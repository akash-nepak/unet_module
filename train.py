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


path_to_data = "/content/drive/MyDrive/ADE20K"

def train(batch_size =32,gradient_accumulation_steps =2,learning_rate= .001,num_epochs =100,image_size = 256,experiment_name = "unet_focal"):
     accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps)


     path_to_experiment = os.path.join("work_dir",experiment_name)

     if not os.path.exists(path_to_experiment): #working directory folder to store checkpints
          os.mkdirs(path_to_experiment)
    
     micro_batchsize = batch_size // gradient_accumulation_steps

     train_data = ADE20KDataset(path_to_data,train=True, image_size=image_size)
     test_data = ADE20KDataset(path_to_data,train=False, image_size=image_size) 
     trainloader = DataLoader(train_data,batch_size=micro_batchsize,shuffle=True,num_workers=8)
     testloader = DataLoader(test_data,batch_size=micro_batchsize,shuffle = True,num_workers =8)


     loss_fn = nn.CrossEntropyLoss(ignore_index =-1) #ignores background information

     model = UNET(in_channels=3,num_classes=150,start_dim=64,dim_mults=(1,2,4,8))

     optimizer = optim.Adam(model.parameters(),lr =learning_rate)

     model,optimizer,trainloader,testloader = accelerator.prepare(  model,optimizer,trainloader,testloader) #for parallel gpu 
    
     
     for epoch in range(1, num_epochs +1):

          accelerator.print(f"Training Epoch[{epoch}/{num_epochs}]")

          train_loss ,test_loss = [],[]
          train_acc, test_acc = [], []
          

          accumulated_loss =0
          accumulated_accuracy =  0
          progress_bar = tqdm(range(len(trainloader)//gradient_accumulation_steps),disable=not accelerator.is_main_process)



          model.train()

          for images,targets in trainloader:

               with accelerator.accumulate(model):
                    pred = model(images)
                    loss = loss_fn(pred,targets)
                    accumulated_loss += loss/gradient_accumulation_steps 

                    predicted = pred.argmax(axis=1) #which pixel has the highest probability along the n channel dimensions 
                    accuracy = (predicted==targets).sum()/torch.numel(predicted)
                    accumulated_accuracy = accuracy/gradient_accumulation_steps

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                         accelerator.clip_grad_norm_(model.parameters(),1.0)

                         loss_gathered = accelerator.gather_for_metrics(accumulated_loss) #accumulate loss from all the gpus

                         accuracy_gathered = accelerator.gather_for_metrics(accumulated_loss)

                         train_loss.append(torch.mean(loss_gathered).item())
                         train_acc.append(torch.mean(accuracy_gathered).item())

                         accumulated_accuracy, accumulated_loss = 0,0 
                         progress_bar.update(1)

                         optimizer.step()
                         optimizer.zero_grad()


          model.eval()

          for images,targets in testloader:

               with torch.no_grad():
                    pred = model(images)

                    loss = loss_fn(pred,targets)

                    predicted = pred.argmax(axis=1)
                    accuracy = (predicted == targets).float().mean()

                    loss_gathered = accelerator.gather_for_metrics(loss)
                    accuracy_gathered = accelerator.gather_for_metrics(accuracy)


                    test_loss.append(torch.mean(loss_gathered).item())
                    test_acc.append(torch.mean(accuracy_gathered).item())


                    epoch_train_loss = np.mean(train_loss)
                    epoch_test_loss = np.mean(test_loss)
                    epoch_train_acc = np.mean(train_acc)
                    epoch_test_acc = np.mean(test_acc)


                    accelerator.print(f"Training Accuracy: {epoch_train_acc},Training Loss {epoch_train_loss}")
                    accelerator.print(f"Testing Accuracy: {epoch_test_acc},Training Loss {epoch_test_loss}")

                    output_dir = os.path.join(path_to_experiment,f"checkpoint {epoch}")
                    accelerator.save_model (model,output_dir)

if __name__ == "__main__":

     train()





