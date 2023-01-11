import torch
import torch.nn as nn
 
OPS = {
	're_conv_421' : lambda C_in, C_out, bias: re_conv_421(C_in, C_out, bias),
	're_conv_622' : lambda C_in, C_out, bias: re_conv_622(C_in, C_out, bias),
	're_conv_823' : lambda C_in, C_out, bias: re_conv_823(C_in, C_out, bias),

	'conv_421' : lambda C_in, C_out, bias: conv_421(C_in, C_out, bias),
	'conv_622' : lambda C_in, C_out, bias: conv_622(C_in, C_out, bias),
	'conv_823' : lambda C_in, C_out, bias: conv_823(C_in, C_out, bias),
}

class re_conv_421(nn.Module):

	def __init__(self, C_in, C_out, bias):
		super(re_conv_421, self).__init__()
		self.op = nn.ConvTranspose2d(in_channels=C_in, out_channels=C_out, kernel_size=4, stride=2, padding=1, bias=bias)

	def forward(self, x):
		return self.op(x)

class re_conv_622(nn.Module):

	def __init__(self, C_in, C_out, bias):
		super(re_conv_622, self).__init__()
		self.op = nn.ConvTranspose2d(in_channels=C_in, out_channels=C_out, kernel_size=6, stride=2, padding=2, bias=bias)

	def forward(self, x):
		return self.op(x)

class re_conv_823(nn.Module):

	def __init__(self, C_in, C_out, bias):
		super(re_conv_823, self).__init__()
		self.op = nn.ConvTranspose2d(in_channels=C_in, out_channels=C_out, kernel_size=8, stride=2, padding=3, bias=bias)

	def forward(self, x):
		return self.op(x)

class conv_421(nn.Module):

	def __init__(self, C_in, C_out, bias):
		super(conv_421, self).__init__()
		self.op = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=4, stride=2, padding=1, bias=bias)

	def forward(self, x):
		return self.op(x)

class conv_622(nn.Module):

	def __init__(self, C_in, C_out, bias):
		super(conv_622, self).__init__()
		self.op = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=6, stride=2, padding=2, bias=bias)

	def forward(self, x):
		return self.op(x)

class conv_823(nn.Module):

	def __init__(self, C_in, C_out, bias):
		super(conv_823, self).__init__()
		self.op = nn.Conv2d(in_channels=C_in, out_channels=C_out, kernel_size=8, stride=2, padding=3, bias=bias)

	def forward(self, x):
		return self.op(x)

class Identity(nn.Module):

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class Zero(nn.Module):

	def __init__(self, stride):
		super(Zero, self).__init__()
		self.stride = stride

	def forward(self, x):
		if self.stride == 1:
			return x.mul(0.)
		return x[:,:,::self.stride,::self.stride].mul(0.)


