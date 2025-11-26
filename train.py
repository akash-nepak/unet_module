import os
import random
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.functional as F
from torch import optim
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup


from dataloader import ADE20KDataset
from model import UNET


path_to_data = "/home/akash-1/train_data/ADE20K"

def train(batch_size =64,gradient_accumulation_steps =1,learning_rate= .001,num_epochs =100,image_size = 256,experiment_name = "unet_focal"):
     accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps)


     path_to_experiment = 

