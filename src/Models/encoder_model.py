import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, dropout=0.2, input_channel = 1):
        """
        Encoder for 2D CSGNet.
        :param dropout: dropout
        """
        super(Encoder, self).__init__()
        self.p = dropout
        self.conv1 = nn.Conv2d(input_channel, 8, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, 3, padding=(1, 1))
        self.drop = nn.Dropout(dropout)
        self.input_channel = input_channel

    def encode(self, x):
        x = F.max_pool2d(self.drop(F.relu(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.drop(F.relu(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
