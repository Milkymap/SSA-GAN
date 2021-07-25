import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
import torch.nn.DataParallel as DP 
 
from libraries.strategies import * 
from libraries.log import logger 

from datalib.data_holder import DATAHOLDER 
from datalib.data_loader import DATALOADER 

from models.damsm import DAMSM

def train_0(storage, nb_epochs, bt_size, pretrained_model):
	nbgpus = th.cuda.device_count() 
	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	source = DATAHOLDER(path_to_storage=storage, for_train=True, max_len=18, neutral='<###>', shape=(256, 256))
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	
	network = DAMSM(vocab_size=len(source.vocab_mapper), common_space_dim=256)
	if nbgpus > 1:
		network = DP(network, device_ids=list(range(nbgpus)))
	
	network.to(device)
	solver = optim.Adam(network.parameters(), lr=0.002, betas=(0.5, 0.999))
	criterion = nn.CrossEntropyLoss().to(device)

	for epoch_counter in range(nb_epochs):
		for index, (images, captions, lengths) in enumerate(loader.loader):
			
			images = images.to(device)
			captions = captions.to(device)

			labels = th.arange(len(images)).to(device)
			response = network(images, captions, lengths)	
			
			words, sentence, local_features, global_features = response 
			wq_prob, qw_prob = network.local_match_probabilities(words, local_features)
			sq_prob, qs_prob = network.global_match_probabilities(sentence, global_features)

			loss_w1 = criterion(wq_prob, labels) 
			loss_w2 = criterion(qw_prob, labels)
			loss_s1 = criterion(sq_prob, labels)
			loss_s2 = criterion(qs_prob, labels)

			loss_sw = loss_w1 + loss_w2 + loss_s1 + loss_s2

			solver.zero_grad()
			loss_sw.backward()
			solver.step()

			message = (epoch_counter, nb_epochs, index, loss_sw.item())
			logger.debug('[%03d/%03d]:%05d >> Loss : %07.3f ' % message)

		if epoch_counter % 100 == 0:
			th.save(network, f'dump/damsm_{epoch_counter}.th')		
	
	th.save(network, f'dump/damsm_{epoch_counter}.th')

@click.command()
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.option('--pretrained_model', help='path to pretrained damsm model', default='')
@click.pass_context
def standard_training(ctx, storage, nb_epochs, bt_size, pretrained_model):
	train_0(storage, nb_epochs, bt_size, pretrained_model)

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug Flag', default=False)
@click.pass_context
def main_command(ctx, debug):
	if not ctx.invoked_subcommand:
		logger.debug('main command')

main_command.add_command(standard_training)

if __name__ == '__main__':
	main_command(obj={})