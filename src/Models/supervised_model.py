"""
Defines Neural Networks
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from ..utils.generators.mixed_len_generator import Parser, \
    SimulateStack, Draw
from typing import List
from ..utils.grammar import Stack, Mask, ImageStack, Tree, edge_counter
import pdb

torch.set_printoptions(threshold=100000)




class ImitateJoint(nn.Module):
    def __init__(self,
                 hd_sz,
                 input_size,
                 encoder,
                 stack_encoder,
                 mode,
                 num_layers=1,
                 num_draws=None,
                 canvas_shape=[64, 64],
                 unique_draw=None,
                 dropout=0.5,
                 num_GPU = 1):
        """
        Defines RNN structure that takes features encoded by CNN and produces program
        instructions at every time step.
        :param num_draws: Total number of tokens present in the dataset or total number of operations to be predicted + a stop symbol = 400
        :param canvas_shape: canvas shape
        :param dropout: dropout
        :param hd_sz: rnn hidden size
        :param input_size: input_size (CNN feature size) to rnn
        :param encoder: Feature extractor network object
        :param mode: Mode of training, RNN, BDRNN or something else
        :param num_layers: Number of layers to rnn
        :param time_steps: max length of program
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

        #### Create Shape Dict
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
            in_features=self.num_draws + 1, out_features=self.input_op_sz)

        self.rnn = nn.LSTM(
            input_size=self.in_sz + self.input_op_sz,
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
        self.stack_encoder = stack_encoder


    def forward(self, x: List, mode = 0, testing = False):
        # program length in this case is the maximum time step that RNN runs
        # 0 reinforce 1 supervised

        data, program_len, input_op, epoch_n = x
        batch_size = data.size()[0]
        hidden = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
        context = Variable(torch.zeros(1, batch_size, self.hd_sz)).cuda()
        sample = torch.ones(batch_size, 1) * (self.num_draws)
        x_f = self.encoder.encode(data[:, 0:1, :, :])
        x_f = x_f.view(1, batch_size, self.in_sz)
        outputs = []
        samples = 30*torch.ones(batch_size, program_len +1).cuda()

        for timestep in range(0, program_len + 1):
            # X_f is the input to the RNN at every time step along with previous
            # predicted label

            if not testing:
                input_op_rnn = self.relu(
                self.dense_input_op(input_op[:, timestep, :]))
            else:

                #arr = Variable(torch.cuda.FloatTensor(batch_size, self.num_draws + 1).fill_(0).scatter_(
                #    1, torch.cuda.LongTensor(sample).view(-1, 1), 1.0)).cuda()
                arr = F.one_hot(sample.long(), self.num_draws + 1).cuda()
                arr = arr.detach()
                temp_input_op = arr.float()
                input_op_rnn = self.relu(
                    self.dense_input_op(temp_input_op))

            input_op_rnn = input_op_rnn.view(1, batch_size,
                                             self.input_op_sz)

            x_f = x_f.view(1, batch_size, self.in_sz)

            input = torch.cat((x_f, input_op_rnn), 2)

            out, hc = self.rnn(input, (hidden, context))
            hidden = hc[0]
            context = hc[1]

            hd = self.relu(self.dense_fc_1(self.drop(out[0])))
            dense_output = self.dense_output(self.drop(hd))

            dense_output_mask = dense_output

            output = self.logsoftmax(dense_output_mask)

            output_probs = self.softmax(dense_output_mask)
            #neg_entropy += torch.sum((output * output_probs), dim=1).unsqueeze(1)

            # Get samples from output probabs based on epsilon greedy way
            # Epsilon will be reduced to 0 gradually following some schedule
            # for terminals, it always sample itself
            #print (output_probs)


            if (testing) :
                sample = torch.max(output_probs, 1)[1].view(
                                    batch_size, 1)
            else:
                sample = torch.multinomial(output_probs, 1).view(
                                batch_size, 1)


            # Stopping the gradient to flow backward from samples
            sample = sample.detach()
            samples[:, timestep] = sample.squeeze(1)


            outputs.append(output)

            sample = sample.cpu()

            #TODO push into stack only if pop_sym is a non-T and remove non-T in reinforce

        return [outputs, samples, [], [], [], [], [], []]


