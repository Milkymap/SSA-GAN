import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
 
from libraries.strategies import * 
from libraries.log import logger 

from datalib.data_holder import DATAHOLDER 
from datalib.data_loader import DATALOADER 

from models.damsm import *


@click.command()
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.option('--pretrained_model', help='path to pretrained damsm model', default='')
def main_loop(storage, nb_epochs, bt_size, pretrained_model):
	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	source = DATAHOLDER(path_to_storage=storage, max_len=18, neutral='<###>', shape=(256, 256))
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	
	if path.isfile(pretrained_model):
		network = th.load(pretrained_model, map_location=device)
	else:
		network = DAMSM(vocab_size=len(source.vocab_mapper), common_space_dim=256)
		network.to(device)
	
	solver = optim.Adam(network.parameters(), lr=0.002, betas=(0.5, 0.999))
	criterion = nn.CrossEntropyLoss().to(device)
	nb_images = 0
	total_images = len(source)

	for epoch_counter in range(nb_epochs):
		for index, (images, captions, lengths) in enumerate(loader.loader):
			batch_size = images.size(0)
			nb_images = nb_images + batch_size 

			images = images.to(device)
			captions = captions.to(device)

			labels = th.arange(len(images)).to(device)
			response = network(images, captions, lengths)	
			
			words, sentence, local_features, global_features = response 
			wq_prob, qw_prob = local_match_probabilities(words, local_features)
			sq_prob, qs_prob = global_match_probabilities(sentence, global_features)

			loss_w1 = criterion(wq_prob, labels) 
			loss_w2 = criterion(qw_prob, labels)
			loss_s1 = criterion(sq_prob, labels)
			loss_s2 = criterion(qs_prob, labels)

			loss_sw = loss_w1 + loss_w2 + loss_s1 + loss_s2

			solver.zero_grad()
			loss_sw.backward()
			solver.step()

			message = (nb_images, total_images, epoch_counter, nb_epochs, index, loss_sw.item())
			logger.debug('I : [%04d/%04d] | E : [%03d/%03d]:%05d | L : %07.3f ' % message)

		if epoch_counter % 100 == 0:
			th.save(network, f'dump/damsm_{epoch_counter}.th')		
	
	th.save(network, f'dump/damsm_{epoch_counter}.th')

if __name__ == '__main__':
	main_loop()