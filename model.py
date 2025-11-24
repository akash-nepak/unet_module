import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

class ResidualBlock(nn.module):

    def __init__(self,in_channels,out_channels,groupnorm_num_group):
        super().__init__()

        self.groupnorm1  = nn.GroupNorm(groupnorm_num_group,in_channels) #normalizes across the channels for every pixel
        self





