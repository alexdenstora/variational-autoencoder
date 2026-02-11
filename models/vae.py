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

   def forward(self, x):
    #TODO

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
    
   def forward(self, x):
      #TODO

class VAE(nn.Module):

  def __init__(self, block_dims = [16, 32, 64, 128], layers_per_scale=2, image_width=64, bottle=512):
    super().__init__()

    #TODO

  def forward(self, x):
    #TODO

