import torch as th 
import torch.nn as nn 
import torch.nn.functional as F 

class MASK_PREDICTOR(nn.Module):
	def __init__(self, icn, hcn=100, out=1):
		super(MASK_PREDICTOR, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(icn, hcn, 3, 1, 1),
			nn.BatchNorm2d(hcn),
			nn.ReLU(),
			nn.Conv2d(hcn, out, 1, 1, 0),
			nn.Sigmoid()
		)

	def forward(self, X):
		return self.body(X)

class SSCBN(nn.Module):
	def __init__(self, icn, ocn, tdf, hdf): 
		super(SSCBN, self).__init__()
		self.mlp_0 = nn.Sequential(nn.Linear(tdf, hdf), nn.ReLU(), nn.Linear(hdf, icn)) 
		self.mlp_1 = nn.Sequential(nn.Linear(tdf, hdf), nn.ReLU(), nn.Linear(hdf, icn))
		self.norma = nn.BatchNorm2d(icn, affine=False)
		self.convl = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(icn, ocn, 3, 1, 1)
		)

	def forward(self, X, M, T):
		N = self.norma(X)
		G = self.mlp_0(T)[:, :, None, None]  # gamma
		B = self.mlp_1(T)[:, :, None, None]  # beta 
		
		W = G * M  
		B = B * M  
		
		R = W * N + B 
		return self.convl(N + R)

class SSACN(nn.Module):
	def __init__(self, icn, ocn, tdf, hdf, hcn, out, up_sacle=1):
		super(SSACN, self).__init__()
		if up_sacle == 1:
			self.head = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.head = nn.Identity()
		self.conv = nn.Conv2d(icn, ocn, 3, 1, 1) if icn != ocn else nn.Identity()
		self.mask = MASK_PREDICTOR(icn, hcn, out)
		self.cbn1 = SSCBN(icn, ocn, tdf, hdf)
		self.cbn2 = SSCBN(ocn, ocn, tdf, hdf)
		self.parm = nn.Parameter(th.zeros(1))

	def forward(self, X, T):
		T = T.transpose(0, 1)
		U = self.head(X)
		M = self.mask(U)
		R = self.conv(U) + self.parm * self.cbn2(self.cbn1(U, M, T), M, T)
		return R, M 

class GENERATOR(nn.Module):
	def __init__(self, noise_dim):
		super(GENERATOR, self).__init__()
		self.head = nn.Linear(noise_dim, 8192)
		self.body = nn.ModuleList([
			SSACN(512, 512, 256, 256, 100, 1, 0),
			SSACN(512, 512, 256, 256, 100, 1, 1),
			SSACN(512, 512, 256, 256, 100, 1, 1),
			SSACN(512, 512, 256, 256, 100, 1, 1),
			SSACN(512, 256, 256, 256, 100, 1, 1),
			SSACN(256, 128, 256, 256, 100, 1, 1),
			SSACN(128, 	64, 256, 256, 100, 1, 1)	
		])
		self.tail = nn.Sequential(
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 3, 3, 1, 1),
			nn.Tanh()
		)

	def forward(self, Z, T):
		N = Z.shape[0]
		X = th.reshape(self.head(Z), (N, 512, 4, 4))
		A = []
		for SSACN_block in self.body: 
			X, M = SSACN_block(X, T)
			A.append(M)
		return self.tail(X), A 


if __name__ == '__main__':
	Z = th.randn((10, 100))
	T = th.randn((10, 256))
	G = GENERATOR()

	print(G)
	R = G(Z, T)
	print(R[0].shape)
