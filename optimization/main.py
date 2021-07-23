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
from models.attngan_discriminator import DISCRIMINATOR

from os import path, mkdir  

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
		mkdir(images_store)

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
	discriminator_network =  DISCRIMINATOR(icn=3, ndf=64, tdf=256).to(device)

	generator_network.train()
	discriminator_network.train()
	logger.debug('Generator and Discriminator were created')
	# define hyparameters
	lambda_DA = 5 
	
	# define solvers and criterions
	dams_criterion = nn.CrossEntropyLoss().to(device)
	gans_criterion = nn.BCELoss().to(device)
	generator_solver = optim.Adam(discriminator_network.parameters(), lr=2e-4, betas=(0.5, 0.999))
	discriminator_solver = optim.Adam(discriminator_network.parameters(), lr=2e-4, betas=(0.5, 0.999)) 

	# main training loop  
	for epoch_counter in range(nb_epochs):
		for index, (real_images, captions, lengths) in enumerate(loader.loader):
			# size current batch
			bsz = len(real_images)   

			# define real and fake labels
			real_labels = th.ones(bsz).to(device)
			fake_labels = th.zeros(bsz).to(device)

			# move data to target device : gpu or cpu 
			real_images = real_images.to(device)
			captions = captions.to(device)
			labels = th.arange(len(real_images)).to(device)

			words, sentences = dams_network.encode_seq(captions, lengths)	
			words, sentences, words.detach(), sentences.detach()

			#-------------------------#
			# train generator network #
			#-------------------------#

			# synthetize fake real_images
			noise = th.randn((bsz, noise_dim)).to(device)
			fake_images, predicted_masks = generator_network(noise, sentences)

			# image and caption encoding
			local_features, global_features = dams_network.encode_img(fake_images)	
			local_features, global_features = local_features.detach(), global_features.detach() 

			wq_prob, qw_prob = dams_network.local_match_probabilities(words, local_features)
			sq_prob, qs_prob = dams_network.global_match_probabilities(sentences, global_features)

			error_w1 = dams_criterion(wq_prob, labels) 
			error_w2 = dams_criterion(qw_prob, labels)
			error_s1 = dams_criterion(sq_prob, labels)
			error_s2 = dams_criterion(qs_prob, labels)

			error_damsm = error_w1 + error_w2 + error_s1 + error_s2

			fake_images_features = discriminator_network(fake_images)
			fake_images_u_features = discriminator_network.get_logits(fake_images_features)
			fake_images_c_features = discriminator_network.get_logits(fake_images_features, sentences)

			generator_u_error = gans_criterion(fake_images_u_features, real_labels)
			generator_c_error = gans_criterion(fake_images_c_features, real_labels)
			
			# compute the deep attentional multimodal similarity

			generator_error =  lambda_DA * error_damsm + 0.5 * (generator_u_error + generator_c_error)

			# backpropagate the error through the generator and dams network

			generator_solver.zero_grad()
			generator_error.backward()
			generator_solver.step()

			#-----------------------------#
			# train discriminator network #
			#-----------------------------#

			real_images_features = discriminator_network(real_images)
			real_images_u_features = discriminator_network.get_logits(real_images_features)
			real_images_c_features = discriminator_network.get_logits(real_images_features, sentences)

			fake_images_features = discriminator_network(fake_images.detach())
			fake_images_u_features = discriminator_network.get_logits(fake_images_features)
			fake_images_c_features = discriminator_network.get_logits(fake_images_features, sentences)

			discriminator_real_u_error = gans_criterion(real_images_u_features, real_labels)
			discriminator_real_c_error = gans_criterion(real_images_c_features, real_labels)

			discriminator_fake_u_error = gans_criterion(fake_images_u_features, fake_labels)
			discriminator_fake_c_error = gans_criterion(fake_images_c_features, fake_labels)

			discriminator_real_error = discriminator_real_u_error + discriminator_real_c_error
			discriminator_fake_error = discriminator_fake_u_error + discriminator_fake_c_error

			discriminator_error = 0.5 * discriminator_real_error + 0.5 * discriminator_fake_error

			discriminator_solver.zero_grad()
			discriminator_error.backward()
			discriminator_solver.step()

			#---------------------------------------------#
			# debug some infos : epoch counter, loss value#
			#---------------------------------------------#
		
			message = (epoch_counter, nb_epochs, index, generator_error.item(), discriminator_error.item())
			logger.debug('[%03d/%03d]:%05d >> GLoss : %07.3f | DLoss : %07.3f' % message)
			
			if index % 10 == 0:
				descriptions = [ source.map_index2caption(seq) for seq in captions]
				output = snapshot(real_images.cpu(), fake_images.cpu(), descriptions, f'output epoch {epoch_counter:03d}', mean=[0.5], std=[0.5])
				#cv2.imshow(f'###.jpg', output)
				#cv2.waitKey(10)

				cv2.imwrite(path.join(images_store, f'###_{epoch_counter:03d}_{index:03d}.jpg'), output)
				
		# temporary model states
		if epoch_counter % 100 == 0:
			th.save(generator_network, f'dump/generator_{epoch_counter:03d}.th')		
	
	th.save(generator_network, f'dump/generator_{epoch_counter:03d}.th')
	logger.success('End of training ...!')

if __name__ == '__main__':
	main_loop()


