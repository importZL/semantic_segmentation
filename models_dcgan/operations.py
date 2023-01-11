import torch
import torch.nn as nn
 
OPS = {
	're_conv_421' : lambda C_in, C_out: re_conv_421(C_in, C_out),
	're_conv_622' : lambda C_in, C_out: re_conv_622(C_in, C_out),
	're_conv_823' : lambda C_in, C_out: re_conv_823(C_in, C_out),
}

class re_conv_421(nn.Module):

	def __init__(self, C_in, C_out):
		super(re_conv_421, self).__init__()
		self.op = nn.Sequential(
			nn.ConvTranspose2d(in_channels=C_in, out_channels=C_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=C_out),
            nn.ReLU(True),
		)

	def forward(self, x):
		return self.op(x)

class re_conv_622(nn.Module):

	def __init__(self, C_in, C_out):
		super(re_conv_622, self).__init__()
		self.op = nn.Sequential(
			nn.ConvTranspose2d(in_channels=C_in, out_channels=C_out, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm2d(num_features=C_out),
            nn.ReLU(True),
		)

	def forward(self, x):
		return self.op(x)

class re_conv_823(nn.Module):

	def __init__(self, C_in, C_out):
		super(re_conv_823, self).__init__()
		self.op = nn.Sequential(
			nn.ConvTranspose2d(in_channels=C_in, out_channels=C_out, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm2d(num_features=C_out),
            nn.ReLU(True),
		)

	def forward(self, x):
		return self.op(x)

