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
                    self.encoder_config.append(((d,channel_sizes[idx+1]),"residual")) #increasing the number of convolutions per layer 

          self.bottleneck_config =[]
          for _ in range(residual_blocks_per_group):
               self.bottleneck_config.append(((ending_channel_size,ending_channel_size),"residual"))

          out_dim = ending_channel_size #512 starting of the bottleneck

          reversed_encoder_config = self.encoder_config[::-1] #making decoder mirror of encoder reversing order

          self.decoder_config =[]
          for idx, (metadata,type) in enumerate(reversed_encoder_config): # developing the bottleneck
               
               enc_in_channels, enc_out_channels = metadata #extracts the number of channel from each layer 

               concat_num_channels = out_dim + enc_out_channels #out_dim = output from the current decoder, enc_out_channel = skip connectionss from encoder

               self.decoder_config.append(((concat_num_channels,enc_in_channels),"residual")) #decoder from 768 conactinated channel to 256 channels, reverse encoder

               if type == "downsample": #resizing image to the size before that layer of encoder block
                    self.decoder_config.append(((enc_in_channels,enc_in_channels),"upsample")) #decoder block

               out_dim = enc_in_channels 
            
          concat_num_channels = starting_channel_size * 2
          self.decoder_config.append(((concat_num_channels,starting_channel_size),"residual"))

          ## actual model implementaton

          self.conv_in_proj = nn.Conv2d(self.input_image_channels,starting_channel_size,kernel_size=3,padding="same") 

          self.encoder = nn.ModuleList() #Encoder module having 
          for metadata, type in self.encoder_config:
               if type == "residual":  #encoder has only 2 types downsample and residual
                    in_channels, out_channels = metadata
                    self.encoder.append(ResidualBlock(in_channels,out_channels,groupnorm_num_groups))
               elif type == "downsample": #we use 
                    in_channels,out_channels = metadata
                    self.encoder.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride =2,padding=1))

          self.bottleneck = nn.ModuleList() #just residual block stacked up
          for(in_channels,out_channels), _ in self.bottleneck_config:
               self.bottleneck.append(ResidualBlock(in_channels,out_channels,groupnorm_num_groups))
               

          self.decoder = nn.ModuleList()    #modulr contains object layers and register it as trainable paramanetr
          for metadata , type in self.decoder_config:
               if  type == "residual":
                    in_channels,out_channels =metadata
                    self.decoder.append(ResidualBlock(in_channels,out_channels,groupnorm_num_groups  ))
               elif type  == "upsample":
                    in_channels,out_channels = metadata
                    self.decoder.append(UpsampleBlock(in_channels,out_channels,interpolate =self.interpolate))
                
                    
          self.conv_out_proj = nn.Conv2d(in_channels=starting_channel_size,out_channels=num_classes,kernel_size=1)

     def forward(self,x):
          
          residuals = [ ] # storing the residual connection images
          x = self.conv_in_proj(x)

          residuals.append(x) # stores residuals from first convolution as well

          for module in self.encoder: #go through each module in encoder and save the residual output from each moduel
                x = module(x)
                residuals.append(x)

          
          for module in self.bottleneck: #bottleneck applied to input image x
               x = module(x)

          for module in self.decoder: # looping through the deoder block
               if isinstance (module,ResidualBlock): 
                    residual_tensor = residuals.pop()

                    x = torch.cat([x,residual_tensor],dim =1)
                    x = module (x)

               else:
                    x = module(x)


          x = self.conv_out_proj(x)
          print(x.shape)
                    
             

                      
               
               

          

           





          


                 



          















           









        

        






if __name__ == "__main__":

    rand = torch.randn(4,3,256,256)
    unet = UNET()
    print(unet(rand))
   

    
    

