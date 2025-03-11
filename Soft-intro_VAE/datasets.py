from torch.utils import data
from PIL import Image
import numpy as np
import torch
import glob
import os

class Edge2Shoe(data.Dataset):
	""" Dataloader for Edge2Shoe datasets 
		Note: we resize images (original 256x256) to 128x128 for faster training purpose 
		
		Args: 
			img_dir: path to the dataset

	"""
	def __init__(self, img_dir):
		image_list = []
		for img_file in glob.glob(str(img_dir)+'*'):
			image_list.append(img_file)
		self.image_list = image_list
		
	def __getitem__(self, index):
		image = Image.open(self.image_list[index]).resize((256,128), resample=Image.BILINEAR)
		image = np.asarray(image).transpose(2,0,1).copy()
		image_tensor = torch.from_numpy(image).float()
		edge_tensor = image_tensor[:,:,:128]; rgb_tensor = image_tensor[:,:,128:]
		return edge_tensor, rgb_tensor

	def __len__(self):
		return len(self.image_list)


class Mask2Image(data.Dataset):
	""" Dataloader for Edge2Shoe datasets 
		Note: we resize images (original 256x256) to 128x128 for faster training purpose 
		
		Args: 
			img_dir: path to the dataset

	"""
	def __init__(self, img_dir, num_images=None):
		image_list = self.__list_full_paths(os.path.join(img_dir, "Images"))
		self.image_list = image_list[:num_images]
		self.mask_list = [img.replace("/Images/", "/Masks/").replace('.jpg','_segmentation.png') for img in self.image_list]
  
	def __list_full_paths(self, img_dir):
		image_list = []
		for img_file in os.listdir(img_dir):
			image_list.append(os.path.join(img_dir, img_file))
		return image_list
		
	def __getitem__(self, index):
		image = Image.open(self.image_list[index]).convert('RGB').resize((128,128), resample=Image.BILINEAR)
		image = np.asarray(image).transpose(2,0,1).copy()
		rgb_tensor = torch.from_numpy(image).float()
  
		edge = Image.open(self.mask_list[index]).convert('RGB').resize((128,128), resample=Image.BILINEAR)
		edge = np.asarray(edge).transpose(2,0,1).copy()
		edge_tensor = torch.from_numpy(edge).float()
		return edge_tensor, rgb_tensor

	def __len__(self):
		return len(self.image_list)


if __name__ == '__main__':
	img_dir = '/data2/li/workspace/data/foot-ulcer/train' 
	dataset = Mask2Image(img_dir, num_images=10)
	loader = data.DataLoader(dataset, batch_size=1)
	print(len(loader))

	for idx, data in enumerate(loader):
		edge_tensor, rgb_tensor = data
		print(idx, edge_tensor.shape, rgb_tensor.shape)

