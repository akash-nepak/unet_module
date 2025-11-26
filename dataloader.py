import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class ADE20KDataset(Dataset):

    def __init__(self,path_to_data,image_size =128,random_crop_ratio =(0.8,1),inference_mode =False,train =True):
         
        self.path_to_data = path_to_data
        self.inference_mode = inference_mode
        self.train = train
        self.image_size = image_size
        self.min_ratio, self.max_ratio = random_crop_ratio


        if train:
            split = "training"

        else:
            split = "validation"

        self.path_to_images = os.path.join(self.path_to_data,"images",split)
        self.path_to_annotations = os.path.join(self.path_to_data,"annotations",split)

        self.file_roots = [path.split(".")[0] for path in os.listdir(self.path_to_images)]

        self.resize = transforms.Resize((self.image_size,self.image_size))
        self.normalize = transforms.Normalize(mean =[0.48897059,0.46548275,0.4294],std =[0.22861765 ,0.224,0.22405])
        self.random_resize = transforms.RandomResizedCrop(size = (self.image_size,self.image_size), #instead of taking whole image it takes random crops and resizes to original image 
                                                          scale = (self.min_ratio,self.max_ratio))
        
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.totensor = transforms.ToTensor()  




    def __len__(self):
        return len(self.file_roots)
    
    def __getitem__(self, idx): # gran file path for image and corresponding anno
        file_root = self.file_roots[idx]

        image = os.path.join(self.path_to_images,f"{file_root}.jpg")
        annot = os.path.join(self.path_to_annotations,f"{file_root}.png")

        image = Image.open(image).convert("RGB")
        annot = Image.open(annot)

        print(image)
        print(np.array(annot))
    

        
    




if __name__ == "__main__":

    path = "/home/akash-1/train_data/ADE20K"

    dataset = ADE20KDataset(path)
    dataset[0]



         


    

