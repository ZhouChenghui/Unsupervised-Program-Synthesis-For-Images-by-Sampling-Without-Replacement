import numpy as np
import torch
import pdb
from ..utils.generators.mixed_len_generator import Draw

class Stack:
    def __init__(self, batch_size, max_len = 100, continuous = False, k = 1, unique_draws = None):
        # creating a map between the grammar token and indices
        self.sym2idx = {'EOP': unique_draws.index("EOP"),
                        'E': unique_draws.index("E"),
                        'T': unique_draws.index("T"),
                        'P': unique_draws.index("P"),
                        'EET': unique_draws.index("EET"),
                        '$': unique_draws.index("$"),
                        'S': unique_draws.index("S")}
        # mostly unused, in case of continuous parameters
        if continuous:
            self.sym2idx = {'EOP': 6, 'E': 7, 'T': 8, 'P': 9, 'EET': 10, '$': 11, 'S': 12}


        self.batch_size = batch_size
        # number of samples for the same target images, or number of beams
        self.k = k
        # length of the stack
        self.max_len = max_len
        # L keeps all the pointers to the stack
        self.L = torch.cuda.LongTensor([2 for i in range(batch_size * k)])
        # initialization of the stack
        self.stack = (torch.ones((self.batch_size * k, max_len), dtype=torch.long) * -1).cuda()
        # Add the end $ and beginning S token to the stack
        self.stack[:, 0:2] = torch.cuda.LongTensor([self.sym2idx['$'], self.sym2idx['S']])
        self.b_idx = torch.cuda.LongTensor(np.arange(self.batch_size * k))
    
    def init(self):
        # set the pointer to position 2
        self.L[:] = 2
        # initialize the empty stack with $ and S
        self.stack[:, 0:2] = torch.cuda.LongTensor([self.sym2idx['$'], self.sym2idx['S']])

    def push(self, S):

        zeros_vec = torch.zeros_like(S)
        # check if the token being pushed consists of multiple tokens or just one
        multi_sym = torch.where(S == self.sym2idx['EET'], S ,zeros_vec)
        single_sym = torch.where(S != self.sym2idx['EET'], S, zeros_vec)
        multi_sym = torch.nonzero(multi_sym).squeeze(1)
        single_sym = torch.nonzero(single_sym).squeeze(1)

        #single token is directly put on the stack
        self.stack[ single_sym, self.L[single_sym ] ] = S[single_sym]
        self.L[ single_sym ] += 1
        #multi symbol  TEE will be put on in 3 steps
        self.stack[ multi_sym, self.L[multi_sym ] ] = self.sym2idx['T']
        self.stack[ multi_sym, self.L[multi_sym]+1 ] = self.sym2idx['E']
        self.stack[ multi_sym, self.L[multi_sym ]+2 ] = self.sym2idx['E']
        self.L[ multi_sym ] += 3

    def pop(self):
        # we pop the token at the position of pointer - 1
        self.L -= 1
        # make sure the pointer doesn't go negative
        z = torch.zeros_like(self.L)
        self.L = torch.max(z, self.L)
        return self.stack[ self.b_idx, self.L]

    def rearrange(self, beam_idx):
        # after each round of stochastic beam search, we need to rearrange the stack
        # according to the order of the beams being selected
        self.stack = torch.index_select(self.stack, 0, beam_idx)
        self.L = torch.index_select(self.L, 0, beam_idx)


class Mask:
    def __init__(self, continuous = False, unique_draws = None):
        
        # the token to index map
        sym2idx = {'EOP': unique_draws.index("EOP"),
                        'E': unique_draws.index("E"),
                        'T': unique_draws.index("T"),
                        'P': unique_draws.index("P"),
                        'EET': unique_draws.index("EET"),
                        '$': unique_draws.index("$"),
                        'S': unique_draws.index("S")}

        # mostly unused, for continuous parameters
        if continuous:
            sym2idx = {'EOP': 6, 'E': 7, 'T': 8, 'P': 9, 'EET': 10, '$': 11, 'S': 12}

        # initialize the mask with zero
        self.mask = torch.cuda.FloatTensor(sym2idx['S']+1, sym2idx['S']+1).fill_(0)
        # E --> EET | P
        self.mask[sym2idx['E'],  [sym2idx['EET'], sym2idx['P']]] = 1
        # T --> * | + | -
        self.mask[sym2idx['T'], sym2idx['EOP']-3:sym2idx['EOP']] = 1
        # P --> SHAPE
        self.mask[sym2idx['P'], :sym2idx['EOP']-3 ] = 1
        # $ --> $
        self.mask[sym2idx['$'], sym2idx['$']] = 1
        # S --> E
        self.mask[sym2idx['S'], sym2idx['E']] = 1
        # EOP --> EOP
        self.mask[ np.arange(sym2idx['EOP']), sym2idx['EOP'] ] = 1

        # log version of the mask, 1 becomes 0, 0 becomes -1e15
        self.mask_logP = torch.cuda.FloatTensor(sym2idx['S']+1, sym2idx['S']+1).fill_(-1e15)
        self.mask_logP[sym2idx['E'],  [sym2idx['EET'], sym2idx['P']]] = 0
        self.mask_logP[sym2idx['T'], sym2idx['EOP']-3:sym2idx['EOP']] = 0
        self.mask_logP[sym2idx['P'], :sym2idx['EOP']-3 ] = 0
        self.mask_logP[sym2idx['$'], sym2idx['$']] = 0
        self.mask_logP[sym2idx['S'], sym2idx['E']] = 0
        self.mask_logP[ np.arange(sym2idx['EOP']), sym2idx['EOP'] ] = 0

    
    def get_mask(self, sym):
        # sym is an array of indices indicating the token,
        # output is the rearranged mask rows
        return self.mask[sym]
    
    def get_mask_logP(self, sym):
        # same thing for the log mask
        return self.mask_logP[sym]

class ImageStack:

    def __init__(self, shape_dict, batch_size, max_len, canvas_shape=[64, 64], k = 1, unique_draws = None):
        self.shape_dict = shape_dict
        self.canvas_shape = canvas_shape

        self.sym2idx = {'+': unique_draws.index("+"),
                        '*': unique_draws.index("*"),
                        '-': unique_draws.index("-"),
                        'EOP': unique_draws.index("EOP"),
                        'E': unique_draws.index("E"),
                        'T': unique_draws.index("T"),
                        'P': unique_draws.index("P"),
                        'EET': unique_draws.index("EET"),
                        '$': unique_draws.index("$"),
                        'S': unique_draws.index("S")}
        self.batch_size = batch_size
        self.k = k
        self.max_len = max_len
        self.L = torch.cuda.LongTensor([0 for i in range(batch_size * k)])
        self.batch_image = torch.zeros([self.batch_size * k, self.max_len+1, self.canvas_shape[0], self.canvas_shape[1]]).cuda()

    def op(self, op_sample):

        zeros_vec = torch.zeros_like(op_sample)

        # select all stacks that perform plus
        add_idx = torch.where(op_sample == self.sym2idx["+"], op_sample, zeros_vec)
        add_idx = torch.nonzero(add_idx)
        if len(list(add_idx.size())) > 1:
            add_idx = add_idx.squeeze(1)
        L_add = self.L[add_idx]
        # select the last two images on the stack
        img2 = self.batch_image[add_idx, L_add - 1, :, :]
        img1 = self.batch_image[add_idx, L_add - 2, :, :]
        # add
        self.batch_image[add_idx, L_add - 2, :, :] = torch.max(img1, img2)
        # zero the second image after the operation
        self.batch_image[add_idx, L_add - 1, :, :] = torch.zeros_like(img1)

        # select all stacks that perform minus
        minus_idx = torch.where(op_sample == self.sym2idx["-"], op_sample, zeros_vec)
        minus_idx = torch.nonzero(minus_idx)
        if len(list(minus_idx.size())) > 1:
            minus_idx = minus_idx.squeeze(1)
        L_minus = self.L[minus_idx]
        # select the last two images on the stack
        img2 = self.batch_image[minus_idx, L_minus - 1, :, :]
        img1 = self.batch_image[minus_idx, L_minus - 2, :, :]
        # minus
        self.batch_image[minus_idx, L_minus - 2, :, :] = torch.max(torch.zeros(img1.size()).cuda(), img1 - img2)
        # zero the second image after the operation
        self.batch_image[minus_idx, L_minus - 1, :, :] = torch.zeros_like(img1)

        # select the stacks that perform intersection
        intersect_idx = torch.where(op_sample == self.sym2idx["*"], op_sample, zeros_vec)
        intersect_idx = torch.nonzero(intersect_idx)
        if len(list(intersect_idx.size() )) > 1:
            intersect_idx = intersect_idx.squeeze(1)
        L_intersect = self.L[intersect_idx]
        # select the last two images on the stack
        img2 = self.batch_image[intersect_idx, L_intersect - 1, :, :]
        img1 = self.batch_image[intersect_idx, L_intersect - 2, :, :]
        # intersect
        self.batch_image[intersect_idx, L_intersect - 2, :, :] = img1 * img2
        # zero the second image after the operation
        self.batch_image[intersect_idx, L_intersect - 1, :, :] = torch.zeros_like(img1)

    def push(self, sample):
        # push new images to the stack
        sample = sample.squeeze(1).long()
        zeros_vec = torch.zeros_like(sample)
        shape_ele = torch.where(sample + 1 < self.sym2idx["+"] + 1, sample + 1, zeros_vec)
        shape_idx = torch.nonzero(shape_ele)
        shape_idx = shape_idx.squeeze(1)
        L_shape = self.L[shape_idx]
        self.batch_image[shape_idx, L_shape, :, :] = torch.index_select(self.shape_dict, 0, sample[shape_idx])
        self.L[shape_idx] += 1

        # choose the stacks that are performing operations this round
        op_ele = torch.where(torch.abs(sample - self.sym2idx["*"]) <= 1, sample, zeros_vec)
        torch.cuda.synchronize()
        op_idx = torch.nonzero(op_ele)
        op_idx = op_idx.squeeze(1)
        self.op(op_ele)
        self.L[op_idx] -= 1

        # make sure L is non-negative
        zeros = torch.zeros_like(self.L)
        ones = torch.ones_like(self.L)
        self.L = torch.min(torch.max(zeros, self.L), ones*self.max_len)

        # push unspecified images
        #zeros_vec = torch.zeros_like(sample)
        #EET_ele = torch.where(sample == self.sym2idx["EET"], sample, zeros_vec)
        #EET_idx = torch.nonzero(EET_ele).squeeze(1)
        #L_EET = self.L[EET_idx]
        #self.batch_image[EET_idx, L_EET, :, :] = torch.ones((L_EET.size()[0], *self.canvas_shape))
        #self.batch_image[EET_idx, L_EET + 1, :, :] = torch.ones((L_EET.size()[0], *self.canvas_shape))

    def rearrange(self, beam_idx, n = 0):
        self.L = torch.index_select(self.L, 0, beam_idx)
        self.batch_image = torch.index_select(self.batch_image, 0, beam_idx)


class ContinuousImageStack:

    def __init__(self, batch_size, max_len, canvas_shape=[64, 64]):
        self.canvas_shape = canvas_shape
        self.sym2idx = {"c": 0, "t": 1, "s": 2, "+": 3, "*": 4, "-": 5, 'EOP': 6, 'E': 7, 'T': 8, 'P': 9, 'EET': 10, '$': 11, 'S': 12}
        self.batch_size = batch_size
        self.max_len = max_len
        self.L = torch.cuda.LongTensor([0 for i in range(batch_size)])
        self.batch_image = torch.zeros([self.batch_size, self.max_len, self.canvas_shape[0], self.canvas_shape[1]]).cuda()
        self.draw = Draw()

    def op(self, op_sample):

        zeros_vec = torch.zeros_like(op_sample)
        add_idx = torch.where(op_sample == self.sym2idx["+"], op_sample, zeros_vec)
        add_idx = torch.nonzero(add_idx).squeeze(1)
        L_add = self.L[add_idx]
        img2 = self.batch_image[add_idx, L_add - 1, :, :]
        img1 = self.batch_image[add_idx, L_add - 2, :, :]
        self.batch_image[add_idx, L_add - 2, :, :] = torch.max(img1, img2)
        self.batch_image[add_idx, L_add - 1, :, :] = torch.zeros_like(img1)

        minus_idx = torch.where(op_sample == self.sym2idx["-"], op_sample, zeros_vec)
        minus_idx = torch.nonzero(minus_idx).squeeze(1)
        L_minus = self.L[minus_idx]
        img2 = self.batch_image[minus_idx, L_minus - 1, :, :]
        img1 = self.batch_image[minus_idx, L_minus - 2, :, :]
        self.batch_image[minus_idx, L_minus - 2, :, :] = torch.max(torch.zeros(img1.size()).cuda(), img1 - img2)
        self.batch_image[minus_idx, L_minus - 1, :, :] = torch.zeros_like(img1)

        intersect_idx = torch.where(op_sample == self.sym2idx["*"], op_sample, zeros_vec)
        intersect_idx = torch.nonzero(intersect_idx).squeeze(1)
        L_intersect = self.L[intersect_idx]
        img2 = self.batch_image[intersect_idx, L_intersect - 1, :, :]
        img1 = self.batch_image[intersect_idx, L_intersect - 2, :, :]
        self.batch_image[intersect_idx, L_intersect - 2, :, :] = img1 * img2
        self.batch_image[intersect_idx, L_intersect - 1, :, :] = torch.zeros_like(img1)

    def push(self, sample, params):
        # push new images to the stack
        sample = sample.squeeze(1).long()
        zeros_vec = torch.zeros_like(sample)
        # circles
        c_ele = torch.where(sample + 1 == self.sym2idx["c"] + 1, sample + 1, zeros_vec)
        c_idx = torch.nonzero(c_ele).squeeze(1)
        L_c = self.L[c_idx]
        self.batch_image[c_idx, L_c, :, :] = self.draw_circle(params[c_idx, :])
        self.L[c_idx] += 1
        # triangles
        t_ele = torch.where(sample + 1 == self.sym2idx["t"] + 1, sample + 1, zeros_vec)
        t_idx = torch.nonzero(t_ele).squeeze(1)
        L_t = self.L[t_idx]
        self.batch_image[t_idx, L_t, :, :] = self.draw_triangle(params[t_idx, :])
        self.L[t_idx] += 1
        # squares
        s_ele = torch.where(sample + 1 == self.sym2idx["s"] + 1, sample + 1, zeros_vec)
        s_idx = torch.nonzero(s_ele).squeeze(1)
        L_s = self.L[s_idx]
        self.batch_image[s_idx, L_s, :, :] = self.draw_square(params[s_idx, :])
        self.L[s_idx] += 1


        # perform operations
        op_ele = torch.where(torch.abs(sample - self.sym2idx["*"]) <= 1, sample, zeros_vec)
        op_idx = torch.nonzero(op_ele).squeeze(1)
        self.op(op_ele)
        self.L[op_idx] -= 1

        # push unspecified images
        #zeros_vec = torch.zeros_like(sample)
        #EET_ele = torch.where(sample == self.sym2idx["EET"], sample, zeros_vec)
        #EET_idx = torch.nonzero(EET_ele).squeeze(1)
        #L_EET = self.L[EET_idx]
        #self.batch_image[EET_idx, L_EET, :, :] = torch.ones((L_EET.size()[0], *self.canvas_shape))
        #self.batch_image[EET_idx, L_EET + 1, :, :] = torch.ones((L_EET.size()[0], *self.canvas_shape))

    def draw_circle(self, params):
        params = params.cpu().numpy()
        num = params.shape[0]
        circles = np.zeros((num, *self.canvas_shape))
        for i in range(num):
            circles[i, :, :] = self.draw.draw_circle([params[i, 0], params[i, 1]], params[i, 2])
        return torch.from_numpy(circles).type(torch.cuda.FloatTensor)

    def draw_square(self, params):
        params = params.cpu().numpy()
        num = params.shape[0]
        squares = np.zeros((num, *self.canvas_shape))
        for i in range(num):
            squares[i, :, :] = self.draw.draw_square([params[i, 0], params[i, 1]], params[i, 2])
        return torch.from_numpy(squares).type(torch.cuda.FloatTensor)

    def draw_triangle(self, params):
        params = params.cpu().numpy()
        num = params.shape[0]
        triangles = np.zeros((num, *self.canvas_shape))
        for i in range(num):
            triangles[i, :, :] = self.draw.draw_triangle([params[i, 0], params[i, 1]], params[i, 2])
        return torch.from_numpy(triangles).type(torch.cuda.FloatTensor)


class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self.parent = parent

class Tree:
    def __init__(self, root, num_branch):
        self.root = Node(root)
        self.pointers = [self.root for i in range(num_branch)]

    def update(self, sample):
        actions = sample.cpu().numpy().flatten().tolist()
        action_included = []
        current_pointer = []
        for ind in range(len(actions)):
            if actions[ind] not in action_included:
                new_node = Node(actions[ind])
                new_node.add_parent(self.pointers[ind])
                self.pointers[ind].add_child(new_node)
                current_pointer.append(new_node)
                action_included.append(actions[ind])
            else:
                other_ind = action_included.index(actions[ind])
                current_pointer.append(current_pointer[other_ind])
        self.pointers = current_pointer

def edge_counter(node):
    num_edges = len(node.children)
    for child in node.children:
        num_edges += edge_counter(child)
    return num_edges