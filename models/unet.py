import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
	
    
		def __init__( self,  n_channels_x=1 ,  bilinear=False ):
			
			super().__init__()
			self.n_channels_x = n_channels_x
			self.bilinear = bilinear
		
			# input  (batch, n_channels_x, 180, 360)
			
			self.initial_conv = InitialConv(n_channels_x, 16)
			# downsampling:
			self.d1 = Down(16, 32)
			self.d2 = Down(32, 64)
			self.d3 = Down(64, 128)
			self.d4 = Down(128, 256)
			self.d5 = Down(256, 512)


			# last conv of downsampling
			self.last_conv_down = DoubleConv(512, 1024)

			# upsampling:

			self.up1 = Up(1024, 512, bilinear= self.bilinear, up_kernel = 2)
			self.up2 = Up(512, 256,bilinear=  self.bilinear, up_kernel = 2)
			self.up3 = Up(256, 128, bilinear= self.bilinear, up_kernel = 2)
			self.up4 = Up(128, 64, bilinear=self.bilinear, up_kernel = 2)
			self.up5 = Up(64, 32, bilinear=self.bilinear, up_kernel = 2)

			# self last layer:
			self.last_conv = OutConv(32, 1)
				

		def forward(self, x):
        # input  (batch, n_channels_x, 40, 360)
			if (type(x) == list) or (type(x) == tuple):    
				x_in = torch.cat([x[0], x[1]], dim=1)
			else:
				x_in = x
			x1 = self.initial_conv(x_in)  # (batch, 16, 180, 360)

		# Downsampling
			x2, x2_bm = self.d1(x1)  # (batch, 32, 90, 180)
			x3, x3_bm = self.d2(x2)  # (batch, 64, 45, 90)
			x4, x4_bm = self.d3(x3)  # (batch, 128, 22, 45)
			x5, x5_bm = self.d4(x4)  # (batch, 256, 11, 22)
			x6, x6_bm = self.d5(x5)  # (batch, 512, 5, 11)

			x7 = self.last_conv_down(x6)  # (batch, 1024, 5, 11)
			
			# Upsampling
			x = self.up1(x7, x6_bm)  # (batch, 512, 11, 22)
			x = self.up2(x, x5_bm)  # (batch, 256, 22, 45)
			x = self.up3(x, x4_bm)  # (batch, 128, 45, 90)
			x = self.up4(x, x3_bm)  # (batch, 64, 90, 180)
			x = self.up5(x, x2_bm)  # (batch, 32, 180, 360)

			x = self.last_conv(x)   # (batch, 1, 180, 360)
			
			return x
    
    

class DoubleConv(nn.Module):
		"""(convolution => [BN] => ReLU) * 2"""
	
		def __init__(self, in_channels, out_channels, mid_channels=None):
				super().__init__()
				if not mid_channels:
						mid_channels = out_channels
				self.double_conv = nn.Sequential(
						nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
						nn.BatchNorm2d(mid_channels),
						nn.ReLU(inplace=True),
						nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
			
		def forward(self, x):
				return self.double_conv(x)
              
class Down(nn.Module):
		"""Downscaling with double conv then maxpool"""
	
		def __init__(self, in_channels, out_channels, pool_padding = 0):
				super().__init__()
				self.maxpool = nn.MaxPool2d(2,stride = 2, padding =  pool_padding)
				self.doubleconv = DoubleConv(in_channels, out_channels)
			
		def forward(self, x):
				x1 = self.doubleconv(x)
				x2 = self.maxpool(x1)
				return x2, x1



class Up(nn.Module):
		"""Upscaling then double conv"""
		def __init__(self, in_channels, out_channels, up_kernel = 3, bilinear=False):
				super().__init__()
			
				# if bilinear, use the normal convolutions to reduce the number of channels
				if bilinear:
						self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
				else:
						self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=up_kernel, stride=2)
						self.conv = DoubleConv(in_channels, out_channels)
					
		def forward(self, x1, x2):
				x1 = self.up(x1)
				# input is CHW
				diffY = x2.size()[2] - x1.size()[2]
				diffX = x2.size()[3] - x1.size()[3]
				x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
												diffY // 2, diffY - diffY // 2])
				# if you have padding issues, see
				# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
				# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
				x = torch.cat([x2, x1], dim=1)
				return self.conv(x)
		

class InitialConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super().__init__()

				self.firstconv = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 1), 
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True)
				)
		def forward(self, x):
				x1 = self.firstconv(x)
				return x1


class OutConv(nn.Module):
		def __init__(self, in_channels, out_channels):
				super().__init__()
				self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
			
		def forward(self, x):
				return self.conv(x)
		







