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
    self.norm2d = LayerNorm2d()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2)
    self.activation = nn.GELU()
    

   def forward(self, x):
    #TODO
    x = self.norm2d(x)
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
    self.input_shape = input_shape
    self.bottle_dim = bottle_dim

    self.flat_dim = input_shape[0] * input_shape[1] * input_shape[2]
    self.norm = nn.LayerNorm(self.flat_dim)
    self.lin_means = nn.Linear(self.flat_dim, bottle_dim)
    self.lin_stds = nn.Linear(self.flat_dim, bottle_dim)
    self.lin_return = nn.Linear(bottle_dim, self.flat_dim)

    self.softplus = nn.Softplus()
  
   def sample(self, means, stds):
     #TODO
     noise = torch.randn_like(stds)
     z = means + stds * noise
     return z

   def generateRandomSamples(self, b, device):
     #TODO
     shape = (b, self.bottle_dim)
     noise = torch.randn(shape)
     noise = noise.to(device)
     noise = self.lin_return(noise)
     noise = torch.reshape(noise, (noise.shape[0], *self.input_shape))
     return noise

   def forward(self, x):
     #TODO
     x = torch.flatten(x, start_dim=1)
     x = self.norm(x)

     means = self.lin_means(x)
     stds = self.lin_stds(x) # need to apply softplus & clamp
     stds = self.softplus(stds) # applying softplus
     stds = torch.clamp(stds, 0.0001, 2) # clamping standard deviations

     z = self.sample(means, stds)
     x = self.lin_return(z)
     x = torch.reshape(x, (x.shape[0], *self.input_shape))
     
     return x, means, stds
   
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
    self.encoder = nn.ModuleList()

    self.encoder.append(nn.Conv2d(in_channels=3, out_channels=block_dims[0], kernel_size=3, padding=1))
    for d in range(1, len(block_dims)):
      self.encoder.append(ResidualConvBlock(d=block_dims[d], layers=layers_per_scale))
      
      if d != len(block_dims) - 1: # if not last block
        self.encoder.append(DownsizeBlock(block_dims[d], block_dims[d+1]))
      else: # if last block
        self.encoder.append(DownsizeBlock(block_dims[d], block_dims[d]))

  def forward(self, x):
    #TODO
    for i in self.layers:
      x = i(x)
    return x
    


