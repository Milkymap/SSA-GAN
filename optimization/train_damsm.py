import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

import torch.distributed as td 
import torch.multiprocessing as tm 

from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.utils.data import DistributedSampler as DSP 
 
from libraries.strategies import * 
from libraries.log import logger 

from datalib.data_holder import DATAHOLDER 
from datalib.data_loader import DATALOADER 

from models.damsm import DAMSM

def train_0(storage, nb_epochs, bt_size, pretrained_model):
	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	source = DATAHOLDER(path_to_storage=storage, for_train=True, max_len=18, neutral='<###>', shape=(256, 256), nb_items=1024)
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	
	if pretrained_model != '' and path.isfile(pretrained_model):
		network = th.load(pretrained_model, map_location=th.device('cpu')).to(device)
	else:
		network = DAMSM(vocab_size=len(source.vocab_mapper), common_space_dim=256).to(device)
	
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
			th.save(network, 'dump/damsm.th')		
	
	th.save(network, 'dump/damsm.th')


def train_1(gpu_id, node_id, nb_gpus_per_node, world_size, server_config, storage, nb_epochs, bt_size):
	worker_rank = node_id * nb_gpus_per_node + gpu_id
	td.init_process_group(backend='nccl', init_method=server_config, world_size=world_size, rank=worker_rank)
	
	th.set_manual_seed(0)
	th.cuda.set_device(gpu_id)

	source = DATAHOLDER(path_to_storage=storage, for_train=True, max_len=18, neutral='<###>', shape=(256, 256), nb_items=1024)
	picker = DSP(dataset=source, num_replicas=world_size, rank=worker_rank)
	loader = DATALOADER(dataset=source, shuffle=False, batch_size=bt_size, sampler=picker)
	
	network = DAMSM(vocab_size=len(source.vocab_mapper), common_space_dim=256).cuda(gpu_id)
	network = DDP(module=network, device_ids=[gpu_id], broadcast_buffers=False)

	solver = optim.Adam(network.parameters(), lr=0.002, betas=(0.5, 0.999))
	criterion = nn.CrossEntropyLoss().cuda(gpu_id)

	for epoch_counter in range(nb_epochs):
		for index, (images, captions, lengths) in enumerate(loader.loader):
			
			images = images.cuda(gpu_id)
			captions = captions.cuda(gpu_id)

			labels = th.arange(len(images)).cuda(gpu_id)
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

			message = (worker_rank, epoch_counter, nb_epochs, index, loss_sw.item())
			logger.debug('(%03d) [%03d/%03d]:%05d >> Loss : %07.3f ' % message)

		if epoch_counter % 100 == 0 and worker_rank == 0:
			th.save(network, 'dump/damsm.th')		
	
	if worker_rank == 0:
		th.save(network, 'dump/damsm.th')

@click.command()
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.option('--pretrained_model', help='path to pretrained damsm model', default='')
@click.pass_context
def standard_training(ctx, storage, nb_epochs, bt_size, pretrained_model):
	train_0(storage, nb_epochs, bt_size, pretrained_model)


@click.command()
@click.option('--cluster_size', help='number of nodes', type=int, default=1)
@click.option('--node_id', help='global id of the host', type=int)
@click.option('--nb_gpus_per_node', help='number of gpu per nodes', type=int)
@click.option('--server_config', help='server tcp address', default='tcp://localhost:8000')
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.pass_context
def parallel_training(ctx, cluster_size, node_id, nb_gpus_per_node, server_config, storage, nb_epochs, bt_size):
	if th.cuda.is_available():
		logger.debug('parallel training is set up')
		world_size = cluster_size * nb_gpus_per_node
		tm.spawn(
			fn=train_1, 
			nprocs=nb_gpus_per_node, 
			args=(node_id, nb_gpus_per_node, world_size, server_config, storage, nb_epochs, bt_size)
		)
	else:
		logger.error('please, use the sub_command standard_training which support single gpu or cpu')


@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug Flag', default=False)
@click.pass_context
def main_command(ctx, debug):
	if not ctx.invoked_subcommand:
		logger.debug('main command')

main_command.add_command(standard_training)
main_command.add_command(parallel_training)

if __name__ == '__main__':
	main_command(obj={})