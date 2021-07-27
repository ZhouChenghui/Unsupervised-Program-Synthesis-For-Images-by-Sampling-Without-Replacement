"""
Defines Neural Networks
"""
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from ..utils.generators.mixed_len_generator import Draw
from typing import List
from ..utils.grammar import Stack, Mask, ImageStack
from src.utils.Gumbel import *
import pdb

torch.set_printoptions(threshold=100000)


class ImitateJoint(nn.Module):
    def __init__(self,
                 hd_sz,
                 input_size,
                 encoder,
                 stack_encoder,
                 mode, # train: 2, test: 1
                 num_layers=1,
                 num_draws=None,
                 canvas_shape=[64, 64],
                 unique_draw=None,
                 dropout=0.5,
                 num_GPU=1):
        """
        Defines RNN structure that takes features encoded by CNN and produces program
        instructions at every time step.
        :param hd_sz: rnn hidden size
        :param input_size: input_size (CNN feature size) to rnn
        :param encoder: Feature extractor network object
        :param stack encoder: Similar to the encoder but for the image stack, thus encode multiple images at once
        :param mode: Either training (2) or testing (1)
        :param num_layers: Number of layers to rnn
        :param num_draws: Total number of tokens present in the dataset + grammatical tokens = total number of items in
        the grammar file
        :param canvas_shape: canvas shape
        :param dropout: dropout
        :param num_GPU: number of GPU for parallelization
        """
        super(ImitateJoint, self).__init__()
        self.hd_sz = hd_sz
        self.in_sz = input_size
        self.num_layers = num_layers
        self.encoder = encoder
        self.mode = mode
        self.canvas_shape = canvas_shape
        self.num_draws = num_draws
        self.unique_draw = unique_draw
        self.init_exp = 0.2
        self.mode = mode

        #### Create Shape Dict ####
        #### Each index correspond to a shape, for image stack ####
        draw = Draw()
        shape_dict = []
        for shape_id in range(len(unique_draw) - 10):
            close_paren = unique_draw[shape_id].index(")")
            value = unique_draw[shape_id][2:close_paren].split(",")
            if unique_draw[shape_id][0] == "c":
                shape_arr = draw.draw_circle([int(value[0]), int(value[1])], int(value[2]))
            elif unique_draw[shape_id][0] == "t":
                shape_arr = draw.draw_triangle([int(value[0]), int(value[1])], int(value[2]))
            elif unique_draw[shape_id][0] == "s":
                shape_arr = draw.draw_square([int(value[0]), int(value[1])], int(value[2]))
            shape_dict.append(shape_arr)
        shape_dict = np.array(shape_dict)
        self.shape_dict = torch.from_numpy(1 * shape_dict).type(torch.FloatTensor)

        # Dense layer to project input ops(labels) to input of rnn
        self.input_op_sz = 128
        self.num_GPU = num_GPU
        self.dense_input_op = nn.Linear(
            in_features=self.num_draws, out_features=self.input_op_sz)

        self.rnn = nn.LSTM(
            input_size=self.in_sz * 2  + self.input_op_sz,
            hidden_size=self.hd_sz,
            num_layers=self.num_layers,
            batch_first=False)

        # adapt logsoftmax and softmax for different versions of pytorch
        self.pytorch_version = torch.__version__[2]

        self.logsoftmax = nn.LogSoftmax(1)
        self.softmax = nn.Softmax(1)
        self.dense_fc_1 = nn.Linear(
            in_features=self.hd_sz, out_features=self.hd_sz)
        self.dense_output = nn.Linear(
            in_features=self.hd_sz, out_features=(self.num_draws))
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batch_norm_emb = nn.BatchNorm1d(self.input_op_sz)
        # Initialize a stack encoder
        self.stack_encoder = stack_encoder

    def forward(self, x: List, k = 1):
        # program length in this case is the maximum time step that RNN runs

        data, program_len, epoch_n = x

        batch_size = data.size()[0]
        # Initialize a grammar stack
        stack = Stack(int(batch_size), k = k, unique_draws=self.unique_draw)
        # Initialize a mask for tree LSTM
        mask = Mask(unique_draws=self.unique_draw)
        # Initialize an image stack
        imgS = ImageStack(self.shape_dict.cuda(), int(batch_size), program_len//4, k = k, unique_draws=self.unique_draw)
        # Initialize inputs to the LSTM
        hidden = Variable(torch.zeros(1, batch_size * k, self.hd_sz)).cuda() #[ for i in range(3 * program_len + 1)]
        context = Variable(torch.zeros(1, batch_size * k, self.hd_sz)).cuda() #[ for i in range(3 * program_len + 1)]
        # Encode the input target images
        x_f = self.encoder.encode(data[:, 0:1, :, :])
        x_f = x_f.view(1, batch_size, self.in_sz).repeat(1, k, 1)
        # Initialize the variables
        outputs = torch.zeros(batch_size*k, program_len).cuda()
        samples = torch.zeros(batch_size*k, program_len).cuda()
        # Initialize the sampling without replacement variables G, logp and entropy as zeros
        G = torch.zeros(batch_size * k, 1).cuda()
        logp = torch.zeros(batch_size * k, 1).cuda()
        entropy_element = torch.zeros(batch_size).cuda()
        ent_plot = torch.zeros(batch_size).cuda()
        stack.init()

        for timestep in range(0, program_len):


            pop_sym = stack.pop()

            # tree lstm
            arr = Variable(
                torch.cuda.FloatTensor(batch_size * k, self.num_draws).fill_(0).scatter_(
                    1, torch.cuda.LongTensor(pop_sym).view(-1, 1), 1.0))
            arr = arr.detach()
            temp_input_op = arr
            input_op_rnn = self.relu(self.dense_input_op(temp_input_op))
            input_op_rnn = input_op_rnn.view(1, batch_size * k,
                                             self.input_op_sz)

            # encode the top-input_channel number of images in the image stack as input to the RNN
            x_f_3 = self.stack_encoder.encode(imgS.batch_image[:, :self.stack_encoder.input_channel, :, :])
            x_f_3 = x_f_3.view(1, batch_size*k, self.in_sz)
            x_f = x_f.view(1, batch_size * k, self.in_sz)

            # construct input to the RNN
            input = torch.cat((x_f, input_op_rnn, x_f_3), 2)

            out, hc = self.rnn(input, (hidden, context))
            hidden = hc[0]
            context = hc[1]

            hd = self.relu(self.dense_fc_1(self.drop(out[0])))
            dense_output = self.dense_output(self.drop(hd))

            dense_output_mask = dense_output + mask.get_mask_logP(pop_sym)
            output_ = self.logsoftmax(dense_output_mask) # batch size by num draw
            output_probs_ = self.softmax(dense_output_mask)
            sample = torch.max(output_probs_, 1)[1].view(batch_size*k, 1)

            if timestep > 0: # this section is the implementation to the stochastic beam search
                
                phi = output_.detach() + logp.detach().repeat((1, self.num_draws))
                g, argmax_g, g_phi = gumbel_with_maximum(phi, G) # g is the stochastic score

                ##### WITH ADJUSTMENTS #####
                g = torch.split(g, batch_size, dim=0)
                ############################

                # How many unique branches there are
                num_unique_branch = timestep if timestep == 1 else num_branch
                g = g[:num_unique_branch]
                g = torch.cat(g, dim=1)

                # How many unique branches there could be expanded from
                num_branch = min(3, k) if timestep == 1 else min(k, torch.min(torch.sum((g > -1e15), 1)).item())

                if self.mode == 2:
                    g_val, index = torch.topk(g, num_branch, dim=1)
                else: # during testing, remove stochasticity
                    g_test = torch.split(phi, batch_size, dim=0)
                    g_test = g_test[:num_unique_branch]
                    g_test = torch.cat(g_test , dim=1)
                    g_val, index = torch.topk(g_test, num_branch, dim=1)
                G[:batch_size*num_branch, :] = torch.cat(torch.split(g_val, 1, dim=1), dim=0)

                # deciding on the beam ID and action ID of the selected branching
                beam_ind = (index / self.num_draws).long()
                action_ind = (index % self.num_draws).long()

                # solidify the sample outputs
                sample_ = torch.split(action_ind, 1, dim=1)
                sample_ = torch.cat(sample_, dim=0)
                sample[:num_branch*batch_size, :] = sample_

                # rearrange the log probability for further branching
                beam_ind = torch.split(beam_ind, 1, dim=1)
                beam_ind = torch.cat(beam_ind, dim=0)
                initial_order = torch.arange(0, batch_size * k).cuda().long()
                order = (beam_ind * batch_size).squeeze(1)  + torch.arange(0, batch_size).repeat((num_branch)).cuda().long()
                initial_order[:batch_size*num_branch] = order
                order = initial_order

                # rearrange grammar stack, image stack, output according the chosen beams
                output = torch.index_select(output_, 0, order)
                output_probs = torch.index_select(output_probs_, 0, order)
                stack.rearrange(order)
                imgS.rearrange(order)
                logp = torch.index_select(logp, 0, order)
                output_mask = torch.cuda.FloatTensor(batch_size*k, self.num_draws).fill_(0).scatter_(
                    1, torch.cuda.LongTensor(sample).view(-1, 1), 1.0)
                logp += torch.sum(output * output_mask, dim = 1).unsqueeze(1)
                pop_sym = torch.index_select(pop_sym, 0, order)

                samples = torch.index_select(samples, 0, order)
                outputs = torch.index_select(outputs, 0, order)


                # Entropy calculation

                ent = torch.sum((output * output_probs) * mask.get_mask(pop_sym), dim=1).unsqueeze(1)

                #calculate weighting for the entropy at each step
                g_k = g_val[:, -1].unsqueeze(1).repeat((1, num_branch-1)).detach()
                phi_k = torch.split(logp, batch_size, dim=0)[:num_branch - 1]
                phi_k = torch.cat(phi_k, dim=1)
                #log_R_s, log_R_ss = compute_log_R(phi_k)
                log_q = gumbel_log_survival(g_k - phi_k.detach())
                log_p = phi_k #
                #log_p_q = log_p + log_R_s
                log_p_q = log_p - log_q
                w_p_q = torch.exp(log_p_q).detach()
                W = torch.sum(w_p_q, dim=1)

                # the k-th beam is used for thresholding
                ent = torch.split(ent, batch_size, dim=0)
                ent = ent[:num_branch - 1]
                ent = torch.cat(ent, dim=1)

                # normalization weights for the entropy
                ent_plot += torch.sum(w_p_q * ent, dim=1)
                ent = torch.sum(w_p_q * ent, dim=1) / W
                entropy_element += ent


                # rnn state rearrange
                hidden[0] = torch.index_select(hidden[0], 0, order)
                context[0] = torch.index_select(context[0], 0, order)

            else:
                output = output_
                output_probs = output_probs_


            # Stopping the gradient to flow backward from samples
            sample = sample.detach()
            samples[:, timestep] = sample.squeeze(1)

            output_mask = torch.cuda.FloatTensor(batch_size * k, self.num_draws).fill_(0).scatter_(
                1, torch.cuda.LongTensor(sample).view(-1, 1), 1.0)

            outputs[:, timestep] = torch.sum((output * output_mask), dim=1)

            zeros_vec = torch.zeros_like(sample).cuda()
            masked_sample = torch.where(sample >= stack.sym2idx['E'], sample, zeros_vec)

            stack.push(masked_sample.squeeze(1))
            imgS.push(sample)
            sample = sample.cpu()
        
        # Rearrange for the parallelization
        samples = torch.cat(torch.split(samples, batch_size, dim=0)[:-1], dim=1)
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0)[:-1], dim=1)
        stack_imgs =  torch.cat(torch.split(imgS.batch_image[:, 0, :, :], batch_size, dim=0)[:-1], dim=1)

        zeros_vec = torch.zeros_like(imgS.L).cuda()
        ones_vec = torch.ones_like(imgS.L).cuda()
        gram_s = torch.where(imgS.L > 1, ones_vec, zeros_vec)
        gram_s = torch.cat(torch.split(gram_s.unsqueeze(1), batch_size, dim=0), dim=1)

        return [outputs, samples, entropy_element, w_p_q, log_p, stack_imgs, gram_s, ent_plot] # ent_plot is the value plotted in tensorbaord and it is unweighted
        #return [outputs, samples, entropy_element, log_R_s, log_R_ss, log_p, stack_imgs, gram_s, ent_plot]



