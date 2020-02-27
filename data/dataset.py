import os
import glob
import torch
import numpy as np
from PIL import Image
import tqdm

import sys

sys.path.append(os.getcwd())

from utils import util


class Dataset(torch.utils.data.Dataset):

	def __init__(self, root):
		super(Dataset, self).__init__()

		# self.root = os.path.join(root, 'frames')
		self.root = root
		self.data = glob.glob('%s/*/*' % root)
		self.labels = os.listdir(root)

	def __getitem__(self, index):
		entry = self.data[index]
		instance = torch.load(entry)
		label = entry.split('/', -1)[-2]

		return instance, int(label)

	def __len__(self):
		return len(self.data)


if __name__ == '__main__':
	testset = Dataset(root='data/something/feats/valid')
	testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

	for idx, batch in enumerate(testloader):
		X, lables = batch
		X = util.batch_cuda(X)
		print(X)

		break
