import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

class ResidualBlock(nn.Module):

    def __init__(self,in_channels,out_channels,groupnorm_num_group):
        super().__init__() #inherits attributes from parent constructor
    

        self.groupnorm1  = nn.GroupNorm(groupnorm_num_group,in_channels) #normalizes across the channels for every pixel
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size= 3, padding = 'same') #padding is same to ensure input image is same dimensions as output image 

        self.groupnorm2 = nn.GroupNorm(groupnorm_num_group,out_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding= 'same') #having same channel dimensions
        
        if in_channels !=out_channels: #handels the shape imbalance issues
            self.residual_match = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:
            self.residual_match = nn.Identity()


    def forward(self,x):

        residual_connection = x # for conacatinating images later 



        x = self.groupnorm1(x)
        x = F.relu(x)
        x = self.conv_1(x)

        x = self.groupnorm2(x)
        x = F.relu(x)
        x = self.conv_2(x)

        x = x + self.residual_match(residual_connection ) #ensuring both x and residual_con have same shape  
        

        return x 
    
class UpsampleBlock(nn.Module):
    def __init__(self,in_channels,out_channels, interpolate = False):
        super().__init__()

        if interpolate :
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                          nn.Conv2d(in_channels,out_channels,kernel_size=3,padding = 'same')) #brute force bilinear interpolation with a convolution operation
        else:
            self.upsample = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2) #causes checkerboard artifacting error

    def forward(self,x):
               return self.upsample(x)



        
class UNET(nn.Module):
     def __init__(self,in_channels=3,num_classes =150,start_dim=64,dim_mults =(1,2,4,8),residual_blocks_per_group =1,groupnorm_num_groups=16,interpolated_upsample =False): #star_dim is input image channel output convolutions
          super().__init__()

          self.input_image_channels = in_channels
          self.interpolate = interpolated_upsample

          channel_sizes = [start_dim * i for i in dim_mults] #tracking change in channel sizes 
          starting_channel_size,ending_channel_size = channel_sizes[0],channel_sizes[-1]

          self.encoder_config =[]

          for idx ,d in enumerate(channel_sizes):
               
               for _ in range (residual_blocks_per_group):#adding no of resudial_block in each channel size 
                    self.encoder_config.append(((d,d),'residual'))

               self.encoder_config.append(((d,d),"downsample"))

               if idx< len(channel_sizes)-1:
                    self.encoder_config.append(((d,channel_sizes[idx+1]),"residual"))

          self.bottleneck_config =[]
          for _ in range(residual_blocks_per_group):
               self.bottleneck_config.append(((ending_channel_size,ending_channel_size),"residual"))

          out_dim = ending_channel_size

          reversed_encoder_config = self.encoder_config[::-1] #making decoder mirror of encoder reversing order

          self.decoder_config =[]
          for idx, (metadata,type) in enumerate(reversed_encoder_config):
               
               enc_in_channels, enc_out_channels = metadata

          















           









        

        






if __name__ == "__main__":

    rand = torch.randn(4,3,256,256)
    unet = UNET()
   

    
    

