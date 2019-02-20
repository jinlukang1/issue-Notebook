import os, sys
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DVSC_dataset(Dataset):
	"""docstring for DVSC_dataloader"""
	def __init__(self, datapath):
		self.datapath = datapath
		self.imgs = [os.path.join(datapath,img) for img in os.listdir(datapath)]

		self.img_transformations = transforms.Compose([
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean = [0.485, 0.456, 0.406], 
									std = [0.229, 0.224, 0.225])
			])

	def __getitem__(self, index):
		img_path = self.imgs[index]

		label = 1 if 'dog' in img_path.split('/')[-1] else 0
		img = Image.open(img_path)
		data = self.img_transformations(img)

		return data, label

	def __len__(self):
		return len(imgs)
		# test_transformations = transforms.Compose([
		# 		transforms.CenterCrop(224),
		# 		transforms.ToTensor(),
		# 		transforms.Normalize(mean = [0.485, 0.456, 0.406], 
		# 							std = [0.229, 0.224, 0.225])
		# 	])
