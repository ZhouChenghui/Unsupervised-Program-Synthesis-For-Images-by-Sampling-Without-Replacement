"""
This file defines helper classes to implement REINFORCE algorithm.
"""
import numpy as np
import torch
from torch.autograd.variable import Variable
from src.utils.parsing import ParseModelOutput, validity
from ..utils.train_utils import chamfer
#from multiprocessing import Pool
from multiprocessing import Process, Queue
import ot
import cv2


class reward_thread(Process):
    def __init__(self,
                 data_dict,
                 unique_draws,
                 canvas_shape,
                 stack_size,
                 k,
                 return_dict,
                 reward="chamfer",
                 if_stack_calculated=False,
                 pred_images=None,
                 power=20):
        """
        This is the thread for the multi threading of the reward generation process.
        :param data_dict: data_dict contains 0: target image;
        :param unique_draws: list of all operation and shape terminals
        :param canvas_shape: canvas shape
        :param stack_size: program_len // 2 + 1
        :param k: number of samples for the same target image
        :param return_dict: the d
        :param reward: The type of reward to specify. The paper uses "chamfer" for cad dataset and
        "chamfer+iou" for synthetic dataset. The rest are for experimentation.
        :param if_stack_calculated: If the final images are already generated from the image stack
        :param pred_images: if_stack_calculated is true, this is the predicted images
        :param power: the power we took of the 1- chamfer distance
        :return: rewards, input samples, predicted images, parsed program expressions, number of invalid programs
        """
        super(reward_thread, self).__init__()
        self.data_dict = data_dict
        self.unique_draws = unique_draws
        self.canvas_shape = canvas_shape
        self.stack_size = stack_size
        self.reward = reward
        self.if_stack_calculated = if_stack_calculated
        self.pred_images = pred_images
        self.power = power
        self.k = k
        self.return_dict = return_dict
        self.daemon = True


    def run(self):

        while True:

            # first get the target images
            self.data = self.data_dict.get()
            if self.if_stack_calculated:
                # if stack calculated the we get the predicted images directly
                pred_images = self.data_dict.get()

            else:
                # samples are the program outputs
                self.samples= self.data_dict.get()


            power = self.power
            num_inval = 0
            if self.reward == "iou":
                power = 1
            if not self.if_stack_calculated:
                # Parse the output indices into program expressions
                parser = ParseModelOutput(self.unique_draws, self.stack_size,[64, 64])

                #samples = np.concatenate(self.samples, 1)
                samples = self.samples.astype(int)

                expressions,_ = parser.labels2exps(samples)

                # Drain all dollars down the toilet!
                for index, exp in enumerate(expressions):
                    expressions[index] = exp.split("$")[0]
                # Generate the predicted images from the program expressions
                pred_images = []
                for index, exp in enumerate(expressions):
                    program = parser.Parser.parse(exp)
                    if validity(program, len(program), len(program) - 1):

                        stack = parser.expression2stack([exp])
                        pred_images.append(stack[0, 0, :, :])
                    else:

                        num_inval += 1
                        pred_images.append(np.zeros(self.canvas_shape))
                pred_images = np.stack(pred_images, 0).astype(dtype=np.bool)

            target_images = self.data[:, 0, :, :].astype(dtype=np.bool)

            image_size = target_images.shape[-1]

            if self.reward == "iou":

                R = np.sum(np.logical_and(target_images, pred_images), (1, 2)) / np.sum(np.logical_or(target_images, pred_images), (1,2))

            elif self.reward == "chamfer":

                distance, r = chamfer(target_images, pred_images)
                # normalize the distance by the diagonal of the image
                R = (1.0 - distance / image_size / (2 ** 0.5))
                R = np.clip(R, a_min=0.0, a_max=1.0)
                R[R > 1.0] = 0
                R = R ** power
                cutoff = np.ones(R.shape) * 0.3
                R = np.maximum(cutoff, R)

            elif self.reward == "chamfer+iou":

                # output the iou and chamfer distance at the same time
                distance, r = chamfer(target_images, pred_images)
                # normalize the distance by the diagonal of the image
                R = (1.0 - distance / image_size / (2 ** 0.5))
                R = np.clip(R, a_min=0.0, a_max=1.0)
                R[R > 1.0] = 0

                R = R ** power
                cutoff = np.ones(R.shape) * 0.3
                R = np.maximum(cutoff, R + r)

            elif self.reward == "distance":

                distance, r = chamfer(target_images, pred_images)
                R = distance

            elif self.reward == "chamfer+iouSolid":

                distance, r = chamfer(target_images, pred_images)
                # normalize the distance by the diagonal of the image
                R = (1.0 - distance / image_size / (2 ** 0.5))
                R = np.clip(R, a_min=0.0, a_max=1.0)
                R[R > 1.0] = 0
                R = R ** power

                intersect = np.sum(np.logical_and(target_images, pred_images) * 1, axis=(1, 2)).astype(np.float)
                union = np.sum(np.logical_or(target_images, pred_images) * 1, axis=(1, 2)).astype(np.float)
                xor = np.sum(np.logical_xor(target_images, pred_images) * 1, axis=(1, 2)).astype(np.float)
                target_sum = np.sum(target_images * 1, axis=(1, 2)).astype(np.float)
                pred_sum = np.sum(pred_images * 1, axis=(1, 2)).astype(np.float)
                r = intersect / target_sum - xor / union

                cutoff = np.ones(R.shape) * 0.3
                R = np.maximum(cutoff, R + r)

            elif self.reward == "OT" or self.reward == "OT_powerup":

                R = []
                for i in range(target_images.shape[0]):
                    img_1 = cv2.Canny(pred_images[i, :, :].astype(np.uint8), 1, 3)
                    img_2 = cv2.Canny(target_images[i, :, :].astype(np.uint8), 1, 3)

                    index_1 = np.array(list(np.where(img_1 == 255)))
                    index_2 = np.array(list(np.where(img_2 == 255)))

                    a, b = np.ones((index_1.shape[1],)) / index_1.shape[1], np.ones((index_2.shape[1],)) / index_2.shape[1]

                    M = ot.dist(index_1.T, index_2.T)
                    M_max = np.amax(M)
                    M = M / M_max
                    dist = ot.emd2(a, b, M)
                    # print (np.allclose(pred_images[i, :, :], target_images[i, :, :]))
                    r = 0
                    # if dist ==    r = 0.5
                    R.append((1 - dist) ** power + r)

            elif self.reward == "binary":
                R = []
                for i in range(target_images.shape[0]):
                    if np.allclose(pred_images[i, :, :], target_images[i, :, :]):
                        R.append(1)
                    else:
                        R.append(0)

            R = np.array(R)
            R = np.expand_dims(R, 1).astype(dtype=np.float32)
            if self.if_stack_calculated:
                self.return_dict.put((R, [], pred_images, [], num_inval))
            else:
                self.return_dict.put((R, samples, pred_images, expressions, num_inval))



class Reinforce:
    def __init__(self,
                 unique_draws,
                 stack_size,
                 reward,
                 power = 20,
                 if_stack_calculated=False,
                 canvas_shape=[64, 64],
                 rolling_average_const= 0.9):

        """
        This class obtains the reward from the output of the network and calculate
        the REINFORCE objectives of the SWOR algorithm
        :param unique_draws: Number of unique_draws in the dataset
        :param stack_size: Maximum size of Stack required
        :param reward: type of reward
        :param power: power of the chamfer distance
        :param if_stack_calculated: if the predicted images are already calculated
        :param canvas_shape: Canvas shape
        :param rolling_average_const: constant to be used in creating running average
        baseline. Only used for naiive REINFORCE
        """
        self.canvas_shape = canvas_shape
        self.unique_draws = unique_draws
        self.max_reward = Variable(torch.zeros(1)).cuda()
        self.rolling_baseline = Variable(torch.zeros(1)).cuda()
        self.alpha_baseline = rolling_average_const
        self.processes = 5
        #manager = Manager()
        self.return_dict = {}
        self.data_dict = {}
        self.if_stack_calculated = if_stack_calculated

        ###### Start the Reward Processes ######
        self.threads = {}

        for k in range(self.processes):
            self.data_dict[k] = Queue(3)
            self.return_dict[k] = Queue(1)
            self.threads[k] = reward_thread(self.data_dict[k],
                 self.unique_draws,
                 self.canvas_shape,
                 stack_size,
                 k,
                 self.return_dict[k],
                 reward=reward,
                 if_stack_calculated=if_stack_calculated,
                 power = power)

            self.threads[k].start()



    def generate_rewards(self,
                         samples,
                         data,
                         images = None,
                         ):
        """
        This function will parse the predictions of RNN into final canvas,
        and define the rewards for individual examples.
        :param samples: Sampled actions from output of RNN
        :param labels: GRound truth labels
        :param power: returns R ** power, to give more emphasis on higher
        powers.
        """
        if self.if_stack_calculated:
            images = images.data.cpu().numpy()
        else:
            images = samples.cpu().numpy() # [step.unsqueeze(1).cpu().numpy() for step in samples]

        batch_size = data.shape[0]
        sub_length = int(batch_size / self.processes)
        extra = batch_size - self.processes * sub_length
        begin = 0
        end = sub_length

        # Manually divide the data evenly by the number of processes
        for key in range(self.processes):
            if extra > 0:
                end += 1
                extra -= 1
                end = min(end, batch_size)

            self.data_dict[key].put(data[ begin:end, :,:,:])
            if self.if_stack_calculated:
                self.data_dict[key].put(images[begin:end, :, :])
            else:
                self.data_dict[key].put(images[begin:end, :]) #[step[begin:end] for step in images])

            begin = end
            end = begin + sub_length

        # putting the data into the dict in the right order in order for the process to .get
        while True:

            S = self.return_dict[0].get()
            R = S[0]
            pred_images = S[2]
            expressions = S[3]
            inval = S[4]
            for k in range(1, self.processes):
                S = self.return_dict[k].get()
                R = np.concatenate((R, S[0]), axis = 0)
                pred_images = np.concatenate((pred_images, S[2]), axis=0)
                expressions += S[3]
                inval += S[4]

            return R, samples, pred_images, expressions, inval/self.processes


    def pg_loss_var(self, R, probs, neg_entropy, param_probs = None, test=False):
        """
        Naive Reinforce loss for variable length program setting, where we stop at maximum
        length programs or when stop symbol is encountered. The baseline is calculated
        using rolling average baseline.
        :return:
        :param R: Rewards for the minibatch
        :param probs: Probability corresponding to every sampled action.
        :param neg_entropy: negative entropy as output from the models
        :param param_probs: contains the probability of the selection at each step. Sum up to get the
        probability of the sequence
        :return loss: reinforce loss
        """
        batch_size = R.shape[0]
        R = Variable(torch.from_numpy(R)).cuda()

        if not test:
            self.rolling_baseline = self.alpha_baseline * self.rolling_baseline + (1 - self.alpha_baseline) * torch.mean(R)
        baseline = self.rolling_baseline.view(1, 1).repeat(batch_size, 1)
        baseline = baseline.detach()
        advantage = R - baseline

        #temp = torch.cat(probs, 1)
        temp = torch.sum(probs, dim=1)

        if param_probs is not None:
            # concatenate along the axis of the same sequence
            param_probs = torch.cat(param_probs, 1)
            # sum up along the same sequence
            temp += torch.sum(param_probs, dim=1)

        loss = -temp.view(batch_size, 1)
        loss = loss.mul(advantage)

        loss = torch.mean(loss)
        loss += torch.mean(neg_entropy)
        #loss += entropy * neg_entropy_coef
        return loss, neg_entropy

    def pg_loss_beam_search(self, R, neg_entropy, w_p_q, log_p, k, test = False):
        """
        Calculate the sampling without replacement REINFORCE objective according to
        the formula in the paper.
        :return:
        :param R: Rewards for the minibatch
        :param neg_entropy: negative entropy as the output of the model
        :param w_p_q: The weighting of the objective formula, details refer to the paper
        :param log_p: log probability of the sequence
        :param k: number of samples for the same target images, or beam size of the stochastic beam search
        :param test: if test time
        :return loss: reinforce loss, full sequence entropy E[log p]
        """

        batch_size = int(R.shape[0]/k)
        R = Variable(torch.from_numpy(R)).cuda()
        R = torch.split(R, batch_size, dim=0)
        R = torch.cat(R, dim=1)
        p_y = torch.exp(log_p)
        W = torch.sum(w_p_q, dim = 1).unsqueeze(1).detach()
        B = torch.sum(w_p_q * R, dim=1).unsqueeze(1).detach()

        W_i = (W.repeat((1, k)) - w_p_q + p_y).detach()
        loss = torch.sum(- w_p_q / W_i * (R - (B/W).repeat(1, k))  * log_p, dim = 1) #
        #loss = torch.sum(- w_p_q * (R ) * log_p, dim=1)
        neg_ent = torch.sum(w_p_q * log_p, dim = 1)

        loss +=  neg_entropy #0.02*neg_ent/ W.squeeze(1) #
        return torch.mean(loss), neg_ent

    def pg_loss_beam_search_rao(self, R, neg_entropy, log_R_s, log_R_ss , log_p, k, test = False):
        """
        Calculate the RAO BLACKWALLized sampling without replacement REINFORCE objective as seen in paper
        Estimating Gradients for Discrete Random Variables by Sampling without Replacement by w. Kool etc.
        This is mostly unused.
        :param R: Rewards for the minibatch
        :param neg_entropy: negative entropy as the output of the model
        :param log_R_s: first set of weightings as output from the model
        :param log_R_ss: second set of weightings as output from the model
        :param log_p: log probability of the sequence
        :param k: number of samples for the same target images, or beam size of the stochastic beam search
        :param test: if test time
        :return loss: reinforce loss, loss without entropy
        """

        batch_size = int(R.shape[0]/k)
        R = Variable(torch.from_numpy(R)).cuda()
        R = torch.split(R, batch_size, dim=0)
        R = torch.cat(R, dim=1)
        log_p_q = log_p + log_R_s.detach()
        w_p_q = torch.exp(log_p_q)
        w_p_R = torch.exp(log_p.unsqueeze(1).repeat((1, k, 1)) + log_R_ss).detach()
        B = torch.sum(w_p_R * R.unsqueeze(1).repeat((1, k, 1)), dim = 2)

        loss = torch.sum(- w_p_q * (R - B), dim = 1)

        reinforce = loss

        loss +=  neg_entropy #0.02*neg_ent/ W.squeeze(1) #

        return torch.mean(loss), torch.mean(reinforce)