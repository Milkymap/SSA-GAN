import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
 
from libraries.strategies import * 
from libraries.log import logger 

from datalib.data_holder import DATAHOLDER 
from datalib.data_loader import DATALOADER 

from models.damsm import DAMSM
from models.generator import GENERATOR
from models.discriminator import DISCRIMINATOR

from os import path 

@click.command()
@click.option('--storage', help='path to dataset: [CUB]')
@click.option('--nb_epochs', help='number of epochs', type=int)
@click.option('--bt_size', help='batch size', type=int)
@click.option('--noise_dim', help='dimension of the noise vector Z', default=100)
@click.option('--pretrained_model', help='path to pretrained damsm model', default='')
@click.option('--images_store', help='generated images will be stored in this directory', default='images_store')
def main_loop(storage, nb_epochs, bt_size, noise_dim, pretrained_model, images_store):
	# intitialization : device, dataset and dataloader 
	device = th.device( 'cuda:0' if th.cuda.is_available() else 'cpu' )
	source = DATAHOLDER(path_to_storage=storage, for_train=True, max_len=18, neutral='<###>', shape=(256, 256), nb_items=1024)
	loader = DATALOADER(dataset=source, shuffle=True, batch_size=bt_size)
	if not path.isdir(images_store):
		os.mkdir(images_store)

	# create networks : dams, generator and discriminator
	if pretrained_model != '' and path.isfile(pretrained_model):
		dams_network = th.load(pretrained_model, map_location=device)
		for p in dams_network.parameters():
			p.requires_grad = False 
		dams_network.eval()
		logger.debug('The pretrained DAMSM was loaded')
	else:
		dams_network = DAMSM(vocab_size=len(source.vocab_mapper), common_space_dim=256).to(device)
	
	generator_network = GENERATOR(noise_dim=noise_dim).to(device) 
	discriminator_network = DISCRIMINATOR(icn=3, ndf=64, tdf=256, min_idx=4, nb_dblocks=6).to(device)

	generator_network.train()
	discriminator_network.train()
	logger.debug('Generator and Discriminator were created')
	# define hyparameters
	p = 6
	lambda_MA = 2
	lambda_DA = 0.1 

	# define solvers and criterions
	dams_criterion = nn.CrossEntropyLoss().to(device)
	generator_solver = optim.Adam(discriminator_network.parameters(), lr=1e-4, betas=(0.0, 0.999))
	discriminator_solver = optim.Adam(discriminator_network.parameters(), lr=4e-4, betas=(0.0, 0.999)) 

	# main training loop  
	for epoch_counter in range(nb_epochs):
		for index, (real_images, captions, lengths) in enumerate(loader.loader):
			# size current batch
			bsz = len(real_images)   

			# move data to target device : gpu or cpu 
			real_images = real_images.to(device)
			captions = captions.to(device)
			labels = th.arange(len(real_images)).to(device)

			#---------------------------------#
			# train generator and dams network# 
			#---------------------------------#
			generator_solver.zero_grad()
			# image and caption encoding
			response = dams_network(real_images, captions, lengths)	
			words, sentences, local_features, global_features = list(map(lambda mdl: mdl.detach(), response)) 
			
			# synthetize fake real_images
			noise = th.randn((bsz, noise_dim)).to(device)
			fake_images, predicted_masks = generator_network(noise, sentences)
			
			# check if fake_real_images is real or not
			fake_images_features = discriminator_network(fake_images, sentences)
			
			# compute the deep attentional multimodal similarity
			wq_prob, qw_prob = dams_network.local_match_probabilities(words, local_features)
			sq_prob, qs_prob = dams_network.global_match_probabilities(sentences, global_features)

			error_w1 = dams_criterion(wq_prob, labels) 
			error_w2 = dams_criterion(qw_prob, labels)
			error_s1 = dams_criterion(sq_prob, labels)
			error_s2 = dams_criterion(qs_prob, labels)

			error_damsm = error_w1 + error_w2 + error_s1 + error_s2
			generator_error = lambda_DA * error_damsm - th.mean(fake_images_features)

			# backpropagate the error through the generator and dams network
			generator_error.backward()
			generator_solver.step()

			#-----------------------------#
			# train discriminator network # 
			#-----------------------------#
			discriminator_solver.zero_grad()
			# extract features of real, fake and mismatch
			real_images_features = discriminator_network(real_images, sentences)
			fake_images_features = discriminator_network(fake_images.detach(), sentences)
			mismatch_images_features = discriminator_network(real_images[:bsz-1], sentences[:, 1:])

			discriminator_error_real = th.mean(th.relu(1 - real_images_features))
			discriminator_error_fake = th.mean(th.relu(1 + fake_images_features))
			discriminator_error_mismatch  = th.mean(th.relu(1 + mismatch_images_features))

			discriminator_error = discriminator_error_real + 0.5 * (discriminator_error_fake + discriminator_error_mismatch)

			# compute the Matching-Aware zero-centered Gradient Penalty 
			interpolated_real_images = (real_images.data).requires_grad_()
			interpolated_sentences = (sentences.data).requires_grad_()
			interpolated_real_images_features = discriminator_network(interpolated_real_images, interpolated_sentences)
			gradients = th.autograd.grad(
				outputs=interpolated_real_images_features, 
				inputs=[interpolated_real_images, interpolated_sentences], 
				grad_outputs=th.ones(interpolated_real_images_features.shape).to(device), 
				only_inputs=True,
				retain_graph=True, 
				create_graph=True 
			)

			interpolated_real_images_gradients = gradients[0]
			interpolated_sentences_gradients = gradients[1]
			
			sentences_norm = th.norm(interpolated_sentences_gradients, p=2, dim=0)
			real_images_norm = th.norm(th.flatten(interpolated_real_images_gradients, start_dim=1), p=2, dim=1)

			MAGP_value = th.mean((sentences_norm + real_images_norm) ** p)
			discriminator_error = discriminator_error + lambda_MA * MAGP_value

			# backpropagate the error through the discriminator network 
			discriminator_error.backward()
			discriminator_solver.step()

			#---------------------------------------------#
			# debug some infos : epoch counter, loss value#
			#---------------------------------------------#
		
			message = (epoch_counter, nb_epochs, index, generator_error.item(), discriminator_error.item())
			logger.debug('[%03d/%03d]:%05d >> GLoss : %07.3f | DLoss : %07.3f' % message)
			
			if index % 2 == 0:
				descriptions = [ source.map_index2caption(seq) for seq in captions]
				output = snapshot(real_images.cpu(), fake_images.cpu(), descriptions, f'output epoch {epoch_counter:03d}', mean=[0.5], std=[0.5])
				cv2.imwrite(path.join(images_store, f'###_{epoch_counter:03d}_{index:03d}.jpg'), output)
				cv2.imwrite(path.join(images_store, f'$$$_{epoch_counter:03d}_{index:03d}.jpg'), th2cv(to_grid(th.cat([real_images.cpu(), fake_images.cpu()], dim=-1))) * 255  )
	
		# temporary model states
		if epoch_counter % 100 == 0:
			th.save(generator_network, f'dump/generator_{epoch_counter:03d}.th')		
	
	th.save(generator_network, f'dump/generator_{epoch_counter:03d}.th')
	logger.success('End of training ...!')

if __name__ == '__main__':
	main_loop()