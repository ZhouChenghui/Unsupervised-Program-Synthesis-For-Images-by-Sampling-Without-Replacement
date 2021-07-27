"""
Training script specially designed for REINFORCE training.
"""

import logging
import numpy as np
import torch
import torch.optim as optim
from src.utils.read_config import Config
from tensorboard_logger import configure, log_value
from torch.autograd.variable import Variable
from src.utils.generators.shapenet_generater import Generator
from src.utils.learn_utils import LearningRate
from src.utils.reinforce_tree import Reinforce
from src.utils.generators.mixed_len_generator import MixedGenerateData, Parser
from torch.nn.parallel import DataParallel
from src.Models.encoder_model import Encoder
from src.Models.tbs_model import ImitateJoint
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='path to config file')

try:
    args = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))

config = Config('config/' + args.config)
manual_seed = 1120
np.random.seed(seed=manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
maxlen = config.max_len*4 + 1
reward = config.reward
power = 20
test_power = 20
DATA_PATH = "data/cad/cad.h5"
model_name = config.model_path.format(config.mode)
config.write_config("log/configs/{}_config.json".format(model_name))

os.makedirs("trained_models/" + str(model_name) + "/", exist_ok=True)
print(config.config)

# Setup Tensorboard logger
configure("log/tensorboard/{}".format(model_name), flush_secs=5)

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(
    'log/logger/{}.log'.format(model_name), mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info(config.config)

# Set the training sizes and testing sizes
# program_len is how long you'd like
dataset_sizes = {

    config.program_len : [config.training_size, config.testing_size]
}
total_keys = len(dataset_sizes.keys())

config.train_size = max(dataset_sizes[k][0] for k in dataset_sizes.keys())
config.test_size = max(dataset_sizes[k][1] for k in dataset_sizes.keys())
total_importance = sum(k for k in dataset_sizes.keys())


# Load the terminals symbols of the grammar
with open(config.grammar_file, "r") as file:
    unique_draw = file.readlines()
for index, e in enumerate(unique_draw):
    unique_draw[index] = e[0:-1]

# Load the cad dataset
if config.data_type == 'cad':
    generator = Generator()

    train_gen = generator.train_gen(
        batch_size=config.gpu_batch, path=DATA_PATH, if_augment=False, shuffle=True)

# Load the synthetic dataset
if config.data_type == 'synthetic':
    data_labels_paths = {  config.program_len : config.data_file }

    generator = MixedGenerateData(
        data_labels_paths=data_labels_paths,
        dataset_sizes=dataset_sizes,
        batch_size=config.gpu_batch,
        test_size=config.test_size,
        num_GPU=config.num_gpu,
        canvas_shape=config.canvas_shape,
        jitter_program=False,
        unique_draw=unique_draw)

    train_gen = generator.get_train_data(config.program_len)


# CNN encoder
encoder_net = Encoder(config.encoder_drop)
stack_encoder = Encoder(input_channel=config.input_channel)

# Initialize the RNN decoder
imitate_net = ImitateJoint(
    hd_sz=config.hidden_size,
    input_size=config.input_size,
    encoder=encoder_net,
    stack_encoder=stack_encoder,
    mode=config.mode,
    num_draws=len(unique_draw),
    unique_draw=unique_draw,
    canvas_shape=config.canvas_shape,
    num_GPU=config.num_gpu)


for param in imitate_net.parameters():
    param.requires_grad = True

for param in encoder_net.parameters():
    param.requires_grad = True

# This defines the reinforce objective
reinforce = Reinforce(stack_size=8,
                      reward=config.reward,
                      power=power,
                      unique_draws=unique_draw,
                      if_stack_calculated=False)


# Optimizers
if config.optim == "sgd":
    optimizer = optim.SGD(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        momentum=0.9,
        lr=config.lr,
        nesterov=False)
elif config.optim == "adam":
    optimizer = optim.Adam(
        [para for para in imitate_net.parameters() if para.requires_grad],
        weight_decay=config.weight_decay,
        lr=config.lr)

# Pre-load the model, including the stack, encoder, main encoder and the RNN's
s_epoch = 0
if config.preload_model:
    print("pre loading model")
    pretrained_dict = torch.load(config.pretrain_modelpath)
    imitate_net_dict = imitate_net.state_dict()
    optimizer.load_state_dict(pretrained_dict['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    pretrained_imi_dict = {
        k: v
        for k, v in pretrained_dict["model_state_dict"].items() if k in imitate_net_dict
    }
    imitate_net_dict.update(pretrained_imi_dict)
    imitate_net.load_state_dict(imitate_net_dict)
    encoder_net.load_state_dict(pretrained_dict['encoder_state_dict'])
    stack_encoder.load_state_dict(pretrained_dict["stack_encoder_state_dict"])
    s_epoch = pretrained_dict['epoch'] + 1

# Learning rate settings
reduce_plat = LearningRate(
    optimizer,
    init_lr=config.lr,
    lr_dacay_fact=0.9,
    patience=config.patience,
    logger=logger)

# Parallel training
imitate_net_para = DataParallel(imitate_net)

imitate_net_para.cuda()
imitate_net_para.train()


for epoch in range(s_epoch, s_epoch + config.epochs + 1):
    print("epoch number ", epoch)
    print("####################")
    train_loss = 0
    total_reward = 0

    for k in dataset_sizes.keys():

        batch_total = config.train_size // config.batch_size
        for batch_idx in range(batch_total):
            print ("#######")
            print (batch_idx)

            loss_sum = Variable(torch.zeros(1)).cuda().data
            Rs = np.zeros((config.gpu_batch, 1))
            neg_entropy_ = 0
            for _ in range(config.batch_size//config.gpu_batch):
                optimizer.zero_grad()
                # Get the data from cad dataset
                data_ = next(train_gen)
                data = Variable(torch.from_numpy(data_), volatile=False).cuda()
                data = data.squeeze(0)
                data_ = np.squeeze(data_, axis=0)

                outputs, samples, neg_entropy, w_p_q, log_p, _, _, ent_plot = imitate_net_para(x=[data, maxlen, epoch],
                                                                                 k=config.beam_n)
                # the output are in the form of batch size by beam size by number of dimension of the values,
                # now we need to reformat them into batch size * beam size by number of dimension of the values
                samples = torch.cat(torch.split(samples, maxlen, dim=1), dim=0)
                outputs = torch.cat(torch.split(outputs, maxlen, dim=1), dim=0)
                data_ = np.tile(data_, (config.beam_n-1, 1, 1, 1))

                # Generates the reward based on image similarity between samples and data_
                R = reinforce.generate_rewards(
                    samples,
                    data_)
                R = R[0]

                # decreasing negative entropy coefficient over time
                neg_entropy_ = 0.999 ** (int(epoch )) * neg_entropy * config.ec
                # the loss function calculated based on the sampling without replacement adjusted objective
                loss, _ = reinforce.pg_loss_beam_search(R, neg_entropy_, w_p_q, log_p, config.beam_n-1)
                #loss, _ = reinforce.pg_loss_beam_search_rao(R, neg_entropy_, log_R_s, log_R_ss, log_p, config.beam_n - 1)
                neg_entropy_ += torch.mean(neg_entropy_)

                # take the max of all the beam's reward
                R = np.vsplit(R, config.beam_n-1)
                R = np.concatenate(R, axis=1)
                R = np.expand_dims(np.max(R, axis=1), axis=1)
                Rs = Rs + R

                loss = loss
                loss.backward()
                loss_sum = loss_sum + loss.data
                l_data = loss.data

                del loss
                # torch.cuda.empty_cache()

            # Clip gradient to avoid explosions
            logger.info(torch.nn.utils.clip_grad_norm(imitate_net_para.parameters(), 10))
            # take gradient step only after having accumulating all gradients.

            l = loss_sum/(config.batch_size/config.gpu_batch)
            R_batch = np.mean(Rs)/(config.batch_size/config.gpu_batch)
            optimizer.step()
            log_value('negative entropy', torch.mean(neg_entropy_/(config.batch_size/config.gpu_batch)).data,
                      epoch * (config.train_size //
                               (config.batch_size)) + batch_idx)
            log_value('train_loss_batch',
                      l.cpu().numpy(),
                      epoch * (config.train_size //
                               (config.batch_size)) + batch_idx)
            log_value('train_reward_batch', R_batch,
                      epoch * (config.train_size //
                               (config.batch_size)) + batch_idx)

            total_reward += R_batch
            train_loss += l

    log_value('train_loss', train_loss/(config.training_size // config.batch_size), epoch)

    log_value('train_reward',
              total_reward / (total_keys * (config.training_size // config.batch_size)), epoch)

    reduce_plat.reduce_on_plateu(-total_reward / (total_keys * (config.training_size // config.batch_size)))
    if epoch % 1 == 0:
        logger.info("Saving the Model weights")
        torch.save({
            'epoch': epoch,
            'model_state_dict': imitate_net.state_dict(),
            "encoder_state_dict": encoder_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "stack_encoder_state_dict":stack_encoder.state_dict(),
            'epoch': epoch
        }, "trained_models/" + str(model_name) + "/" + str(epoch) + ".pth")

for thread in reinforce.threads.values():
    thread.terminate()
