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

def train(batch_size =64,gradient_accumulation_steps =2,learning_rate= .001,num_epochs =100,image_size = 256,experiment_name = "unet_focal"):
     accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps)


     path_to_experiment = os.path.join("work_dir",experiment_name)

     if not os.path.exists(path_to_experiment): #working directory folder to store checkpints
          os.mkdir(path_to_experiment)
    
    micro_batchsize = batch_size // gradient_accumulation_steps

    train_data = ADE20KDataset(path_to_data,train=True, image_size=image_size)
    test_data = ADE20KDataset(path_to_data,train=False, image_size=image_size) 
    trainloader = DataLoader(train_data,batch_size=micro_batchsize,shuffle=True,num_workers=8)
    testloader = DataLoader(test_data,batch_size=micro_batchsize,shuffle = True,num_workers =8)


     loss_fn = nn.

     
    







