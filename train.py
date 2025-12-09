import os
import random
from PIL import Image
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def train(batch_size =128,gradient_accumulation_steps =2,learning_rate= .0001,num_epochs =100,image_size = 256,experiment_name = "unet_focal"):
     accelerator = Accelerator(gradient_accumulation_steps = gradient_accumulation_steps)


     path_to_experiment = os.path.join("work_dir",experiment_name)

     if not os.path.exists(path_to_experiment): #working directory folder to store checkpints
          os.makedirs(path_to_experiment,exist_ok = True)
    
     micro_batchsize = batch_size // gradient_accumulation_steps

     train_data = ADE20KDataset(path_to_data,train=True, image_size=image_size)
     test_data = ADE20KDataset(path_to_data,train=False, image_size=image_size) 
     trainloader = DataLoader(train_data,batch_size=micro_batchsize,shuffle=True,num_workers=4)
     testloader = DataLoader(test_data,batch_size=micro_batchsize,shuffle = False,num_workers =4)


     loss_fn = nn.CrossEntropyLoss(ignore_index =-1) #ignores background information

     model = UNET(in_channels=3,num_classes=150,start_dim=64,dim_mults=(1,2,4,8))

     optimizer = optim.Adam(model.parameters(),lr =learning_rate)
     total_steps = (len(trainloader) * num_epochs) // gradient_accumulation_steps
     scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)
     

     model,optimizer,trainloader,testloader = accelerator.prepare(  model,optimizer,trainloader,testloader) #for parallel gpu 
     best_test_loss = np.inf
     
     for epoch in range(1, num_epochs +1):

          accelerator.print(f"Training Epoch[{epoch}/{num_epochs}]")

          train_loss ,test_loss = [],[]
          train_acc, test_acc = [], []
          

          accumulated_loss =0.0
          accumulated_accuracy  =  0
          progress_bar = tqdm(range(len(trainloader)//gradient_accumulation_steps),disable=not accelerator.is_main_process)



          model.train()

          for images,targets in trainloader:
              with accelerator.accumulate(model):
                    pred = model(images)
                    loss = loss_fn(pred, targets)
                    
                   
                    step_loss = loss / gradient_accumulation_steps
                    
                   
                    accelerator.backward(step_loss)
                    
                    
                    accumulated_loss += step_loss.detach() 

                   
                    predicted = pred.argmax(dim=1) 
                    accuracy = (predicted == targets).float().mean()
                    
                   
                    accumulated_accuracy += accuracy.detach() / gradient_accumulation_steps

                    if accelerator.sync_gradients:
                         accelerator.clip_grad_norm_(model.parameters(), 1.0)

                       
                         loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
                         accuracy_gathered = accelerator.gather_for_metrics(accumulated_accuracy)

                         train_loss.append(torch.mean(loss_gathered).item())
                         train_acc.append(torch.mean(accuracy_gathered).item())

                     
                         accumulated_accuracy = 0.0
                         accumulated_loss = 0.0
                         
                         progress_bar.update(1)
                         progress_bar.set_postfix({"loss": train_loss[-1]})

                         optimizer.step()
                         scheduler.step()
                         optimizer.zero_grad()


          model.eval()

          for images,targets in testloader:

               with torch.no_grad():
                    pred = model(images)

                    loss = loss_fn(pred,targets)

                    predicted = pred.argmax(dim=1)
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


          if epoch_test_loss < best_test_loss: # saving the best model, to prevent model from overfitting
              accelerator.print("--saving--")
              best_test_loss = epoch_test_loss

              best_dir = os.path.join(path_to_experiment,f"best_checkpoint {epoch}")
              if accelerator.is_main_process:
                 os.makedirs(best_dir,exist_ok=True)
                 unwrapped_model = accelerator.unwrap_model(model)
                 accelerator.save(unwrapped_model.state_dict(), os.path.join(best_dir, "model_state.pt"))
                         

          output_dir = os.path.join(path_to_experiment,f"last_checkpoint {epoch}")
          if accelerator.is_main_process:
           os.makedirs(output_dir,exist_ok=True)
           unwrapped_model = accelerator.unwrap_model(model)
           accelerator.save(unwrapped_model.state_dict(), os.path.join(output_dir, "model_state.pt"))
                 

if __name__ == "__main__":

     train()





