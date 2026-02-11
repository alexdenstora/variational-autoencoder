import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class DownsizeBlock(nn.Module):
   def __init__(self, in_channels, out_channels):
    super().__init__()
    #TODO
    # stride should halve the input resolution
    #layernorm2d, Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2), GELU()
    self.norm = LayerNorm2d()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2)
    self.activation = nn.GELU()
    

   def forward(self, x):
    #TODO
    x = self.norm(x)
    x = self.conv(x)
    x = self.activation(x)
    return x

class UpsizeBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    #TODO

  def forward(self, x):
    #TODO


class Bottleneck(nn.Module):
   def __init__(self, input_shape, bottle_dim=64):
    super().__init__()

    #TODO
  
   def sample(self, means, stds):
     #TODO

   def generateRandomSamples(self, b, device):
     #TODO

   def forward(self, x):
     #TODO
   
class ResidualConvBlock(nn.Module):
   def __init__(self, d, layers, width=3):
      super().__init__()
     #TODO
     # 2d convolutional layers w/ GELU's in between
     # pad each convolution so that dimensions do not change
     pad = (width - 1)/2
     self.conv = nn.Conv2d(channels_in=d, channels_out=d, kernel_size=width, padding=pad)
     self.activation = nn.GELU()
     self.layers = layers
    
   def forward(self, x):
      #TODO
      # apply a residual around the entire set of layers
      residual = x
      for i in range(len(self.layers)):
        x = self.conv(x)
        x = self.activation(x)
      return x + residual
      

class VAE(nn.Module):

  def __init__(self, block_dims = [16, 32, 64, 128], layers_per_scale=2, image_width=64, bottle=512):
    super().__init__()
    #TODO

  def forward(self, x):
    #TODO
    


