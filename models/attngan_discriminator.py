import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np
import operator as op 
import itertools as it, functools as ft 

class EXTRACTOR(nn.Module):
	def __init__(self, icn, ocn):
		super(EXTRACTOR, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(icn, ocn, 3, 1, 1, bias=False), 
			nn.BatchNorm2d(ocn),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, X):
		return self.body(X)

class DOWNBLOCK(nn.Module):
	def __init__(self, icn, ocn, bnorm=True):
		super(DOWNBLOCK, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(icn, ocn, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ocn) if bnorm else nn.Identity(),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, X):
		return self.body(X)

class CONCATBLOCK(nn.Module):
	def __init__(self, icn, ocn, tdf):
		super(CONCATBLOCK, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(icn + tdf, ocn, 3, 1, 1, bias=False),
			nn.BatchNorm2d(ocn),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, X, T):    
		_, _, H, W = X.shape
		S = T[:, :, None, None].repeat(1, 1, H, W)
		XS = th.cat((X, S), dim=1)
		return self.body(XS)


class DISCRIMINATOR(nn.Module):
	def __init__(self, icn, ndf, tdf, down_factor=64):
		super(DISCRIMINATOR, self).__init__()
		self.nb_down = int(np.log2(down_factor)) 
		self.head = DOWNBLOCK(icn, ndf, bnorm=False)
		self.body = nn.Sequential(*[
			DOWNBLOCK(ndf * 2 ** idx, ndf * 2 ** (idx + 1))
			for idx in range(self.nb_down - 1)
		])

		self.term = nn.Sequential(
			EXTRACTOR(ndf * 2 ** (self.nb_down - 1), ndf * 2 ** (self.nb_down - 2)),
			EXTRACTOR(ndf * 2 ** (self.nb_down - 2), ndf * 2 ** (self.nb_down - 3)),
		)

		self.unet = nn.Sequential(
			nn.Conv2d(ndf * 2 ** (self.nb_down - 3), 1, 4, 4, 1), 
			nn.Sigmoid()
		) 

		self.ccat = CONCATBLOCK(ndf * 2 ** (self.nb_down - 3), ndf * 2 ** (self.nb_down - 3), tdf)
		self.cnet = nn.Sequential(
			nn.Conv2d(ndf * 2 ** (self.nb_down - 3), 1, 4, 4, 1),
			nn.Sigmoid()
		) 
	
	def get_logits(self, X, T=None):
		if T is None:
			return th.squeeze(self.unet(X))
		else:
			T = T.transpose(0, 1)
			return th.squeeze(self.cnet(self.ccat(X, T)))

	def forward(self, X):
		return self.term(self.body(self.head(X)))

if __name__ == '__main__':
	D = DISCRIMINATOR(3, 64, 256)
	print(D)

	X = th.randn(2, 3, 256, 256)
	Y = D(X)
	print(Y.shape)

	print(D.get_logits(Y).shape)

	T = th.randn(2, 256)

	print(D.get_logits(Y, T).shape)

