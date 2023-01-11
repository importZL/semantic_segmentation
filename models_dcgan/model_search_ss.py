import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_dcgan.operations import *
from torch.autograd import Variable
from models_dcgan.genotypes import PRIMITIVES
from models_dcgan.genotypes import Genotype

 
class MixedOp(nn.Module):

	def __init__(self, C_in, C_out):
		super(MixedOp, self).__init__()
		self._ops = nn.ModuleList()
		for primitive in PRIMITIVES:
			op = OPS[primitive](C_in, C_out)
			self._ops.append(op)

	def forward(self, x, weights):
		return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

	def __init__(self, layers, channel_list):
		super(Cell, self).__init__()
		self._layers = layers
		self._ops = nn.ModuleList()
		for i in range(self._layers):
			C_in, C_out = channel_list[i], channel_list[i+1]
			op = MixedOp(C_in, C_out)
			self._ops.append(op)

	def forward(self, input, weights):
		states = input
		for i in range(self._layers):
			states = self._ops[i](states, weights[i,:])

		return states


class Network(nn.Module):

	def __init__(self, layers, channel_list=None, cuda_index=0):
		super(Network, self).__init__()
		self._channel_list = channel_list
		self._layers = layers
		self._cuda_index = cuda_index

		assert self._layers == len(self._channel_list)-1, 'number of layers do not match the len of channel_list'

		self.main_module = Cell(self._layers, self._channel_list)

		self._initialize_alphas()

	def new(self):
		model_new = Network(self._C, self._layers).cuda(self._cuda_index)
		for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
			x.data.copy_(y.data)
		return model_new

	def forward(self, input):
		
		out = self.main_module(input, F.softmax(self.alphas_normal, dim=-1))
		return out

	def feature_extraction(self, input):
		out = self.main_module(s1, F.softmax(self.alphas_normal, dim=-1))
		return out

	def _initialize_alphas(self):
		k = self._layers
		num_ops = len(PRIMITIVES)

		self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(self._cuda_index), requires_grad=True)
		self._arch_parameters = [
			self.alphas_normal,
		]

	def arch_parameters(self):
		return self._arch_parameters


