import torch as th 

from models.damsm import DAMSM 
from datalib.data_holder import DATAHOLDER
from datalib.data_loader import DATALOADER


from libraries.strategies import * 

if __name__ == '__main__':
	model = th.load('dump/damsm_1024.th', map_location=th.device('cpu')).eval()

	source = DATAHOLDER(path_to_storage='storage', for_train=True, max_len=18, neutral='<###>', shape=(256, 256))
	loader = DATALOADER(dataset=source, shuffle=False, batch_size=8)
	
	img0, cap0, lng0 = source[10]
	vals = source.get_caption(10)

	iccm = [img0]
	cccm = [cap0]
	lccm = [lng0]
	for idx in range(15):
		img1, cap1, lng1 = source[np.random.randint(len(source))]
		
		iccm.append(img1)
		cccm.append(cap0)
		lccm.append(lng0)

	imgs = th.stack(iccm)
	caps = th.stack(cccm)
	lngs = lccm
	
	idxs = th.randperm(len(imgs))
	imgs = imgs[idxs, ...]
	
	with th.no_grad():
		resp = model(imgs, caps, lngs)

		wrds, snts, lfea, gfea = resp 
		wq_prob, qw_prob = model.local_match_probabilities(wrds, lfea)
		sq_prob, qs_prob = model.global_match_probabilities(snts, gfea)

		prbs = th.softmax(qw_prob + qs_prob, dim=1)
		outp = th.argmax(th.diag(prbs))
		
		print(th.diag(prbs))
		print('index of the bird : ',outp.item())
		print(vals)

		grid = to_grid((imgs * 0.5) + 0.5, nb_rows=4)
		cv2.imshow('...', cv2.resize(th2cv(grid), (800, 800)))
		cv2.waitKey(0)
	





