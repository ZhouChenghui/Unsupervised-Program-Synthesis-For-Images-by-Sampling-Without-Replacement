import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd.variable import Variable
from src.utils.generators.mixed_len_generator import Parser, \
    SimulateStack
from typing import List
import pdb


class ParseModelOutput:
    def __init__(self, unique_draws: List, stack_size: int,
                 canvas_shape: List, posx = None, posy = None, size = None):
        """
        This class parses complete output from the network which are in joint
        fashion. This class can be used to generate final canvas and
        expressions.
        :param unique_draws: Unique draw/op operations in the current dataset
        :param stack_size: Stack size
        :param steps: Number of steps in the program
        :param canvas_shape: Shape of the canvases
        """
        self.canvas_shape = canvas_shape
        self.stack_size = stack_size

        self.Parser = Parser()
        self.sim = SimulateStack(self.stack_size, self.canvas_shape)
        self.unique_draws = unique_draws
        # self.pytorch_version = torch.__version__[2]
        if "EOP" in unique_draws:
            self.n_T = unique_draws.index("EOP") # 30  # EOP cannot render an image
        else:
            self.n_T = unique_draws.index("$")
        self.posx = posx
        self.posy = posy
        self.size = size

    def get_final_canvas(self,
                         outputs: List,
                         if_just_expressions=False,
                         if_pred_images=False):
        # TODO
        return

    def expression2stack(self, expressions: List):
        """Assuming all the expression are correct and coming from
        groundtruth labels. Helpful in visualization of programs
        :param expressions: List, each element an expression of program
        """
        stacks = []
        for index, exp in enumerate(expressions):
            program = self.Parser.parse(exp)
            self.sim.generate_stack(program)
            stack = np.array(self.sim.stack_t[-1][0:1])
            # stack = np.stack(stack, axis=0)
            stacks.append(stack)
        stacks = np.stack(stacks, 0).astype(dtype=np.float32)
        return stacks

    def labels2exps(self, labels: np.ndarray):

        """
        Assuming grountruth labels, we want to find expressions for them
        :param labels: Grounth labels batch_size x time_steps
        :return: expressions: Expressions corresponding to labels
        """

        if isinstance(labels, np.ndarray):
            batch_size = labels.shape[0]
        else:
            batch_size = labels.size()[0]
            labels = labels.data.cpu().numpy()
        # Initialize empty expression string, len equal to batch_size
        correct_programs = []
        expressions = [""] * batch_size
        p_len = [5] * batch_size
        pre = 0

        for j in range(batch_size):
            i = 0
            while i < (labels.shape[1]):
                # TODO replace the pre specified value
                if labels[j, i] < self.n_T:

                    if (not self.posx is None) and labels[j, i] < self.unique_draws.index("POSY_1"):
                        if i + 3 < labels.shape[1]:
                            shape = self.unique_draws[labels[j, i]]
                            x_param = self.posx[labels[j, i+1] - self.unique_draws.index("POSX_1") ]
                            y_param = self.posy[labels[j, i + 2] - self.unique_draws.index("POSY_1")]
                            sz_param = self.size[labels[j, i + 3] - self.unique_draws.index("SIZE_1")]
                            expressions[j] += shape + "(" + str(x_param) + "," + str(y_param) + "," + str(sz_param) + ")"
                            i += 3
                        else:
                            expressions[j] += "c(8,8,8)" # throw in a random shape so that it will become an invalid program
                            i = labels.shape[1]

                    else:
                        expressions[j] += self.unique_draws[int(labels[j, i])]

                elif labels[j, i] == self.unique_draws.index('$'):
                    p_len[j] = i
                    break
                i += 1

        return expressions, p_len


def validity(program: List, max_time: int, timestep: int):
    """
    Checks the validity of the program. In short implements a pushdown automaton that accepts valid strings.
    :param program: List of dictionary containing program type and elements
    :param max_time: Max allowed length of program
    :param timestep: Current timestep of the program, or in a sense length of
    program
    # at evey index
    :return:
    """
    num_draws = 0
    num_ops = 0
    for i, p in enumerate(program):
        if p["type"] == "draw":
            # draw a shape on canvas kind of operation
            num_draws += 1
        elif p["type"] == "op":
            # +, *, - kind of operation
            num_ops += 1
        elif p["type"] == "stop":
            # Stop symbol, no need to process further
            if num_draws > ((len(program) - 1) // 2 + 1):
                return False
            if not (num_draws > num_ops):
                return False
            return (num_draws - 1) == num_ops

        if num_draws <= num_ops:
            # condition where number of operands are lesser than 2
            return False
        if num_draws > (max_time // 2 + 1):
            # condition for stack over flow
            return False
    if (max_time - 1) == timestep:
        return (num_draws - 1) == num_ops
    return True