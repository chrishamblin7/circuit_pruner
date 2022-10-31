import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
import numpy as np
import os
from circuit_pruner.utils import load_config
from circuit_pruner.data_loading import rank_image_data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from circuit_pruner.simple_api.mask import mask_from_scores, apply_mask,setup_net_for_mask
from circuit_pruner.simple_api.target import feature_target_saver, sum_abs_loss
from time import time
import pickle


#params
config_file = '../configs/resnet18_config.py'
scores_folder = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/circuit_ranks/resnet18/imagenet_2/actxgrad/'
out_folder = './resnet18/circuit_activations/'
device = 'cuda:0'
batch_size = 64
sparsities = [.5,.4,.3,.2,.1,.05,.01]

#model
config = load_config(config_file)
model = config.model
_ = model.to(device).eval()


#dataloader
kwargs = {'num_workers': 4, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}
dataloader = torch.utils.data.DataLoader(rank_image_data(config.data_path,
										config.preprocess,
										label_file_path = config.label_file_path,class_folders=True),
										batch_size=batch_size,
										shuffle=False,
										**kwargs)


all_correlations = OrderedDict()

#get data
score_files = os.listdir(scores_folder)
for score_file in score_files: 
	start = time()

	#import pdb;pdb.set_trace()
	print(score_file)
	if os.path.exists(out_folder+score_file):
		print('skipping')
		continue
	score_dict = torch.load(os.path.join(scores_folder,score_file))
	unit = score_dict['unit']
	target_layer = score_dict['layer']
	scores = score_dict['scores']

	setup_net_for_mask(model) #reset mask in net
	#original_activations
	original_activations = []
	print('original activations')
	#we save target activations in a context that allows us to handle the annoying problem of dangling hooks
	with feature_target_saver(model,target_layer,unit) as target_saver:
		#then we just run our data through the model, the target_saver will store activations for us
		for i, data in enumerate(dataloader, 0):
			inputs, labels = data
			inputs = inputs.to(device)
			target_activations = target_saver(inputs)
			#the target_saver doesnt aggregate activations, it overwrites each batch, so we need to save our data
			original_activations.append(target_activations.detach().cpu().type(torch.FloatTensor))

		#turn batch-wise list into concatenated tensor
		original_activations = torch.cat(original_activations)
	

	circuit_activations = []
	with feature_target_saver(model,target_layer,unit) as target_saver:
		for sparsity in sparsities:
		#for sparsity in sparsities[cat]:
			print(sparsity)

			#MASK THE MODEL
			mask = mask_from_scores(scores,sparsity = sparsity,model=model,unit=unit,target_layer_name=target_layer)
			apply_mask(model,mask)

			activations = []
			#then we just run our data through the model, the target_saver will store activations for us
			for i, data in enumerate(dataloader, 0):
				model.zero_grad()

				inputs, target = data
				inputs = inputs.to(device)
				target = target.to(device)

				target_activations = target_saver(inputs)
				#import pdb; pdb.set_trace()
				#loss = sum_abs_loss(target_activations)
				#loss.backward()
				
				#save activations
				activations.append(target_activations.detach().cpu().type(torch.FloatTensor))
				

			#turn batch-wise list into concatenated tensor
			activations = torch.cat(activations)
			circuit_activations.append(activations)


	#correlations
	correlations = []
	for a in circuit_activations: 
		correlations.append(np.corrcoef(a.flatten(),original_activations.flatten())[0][1])

	all_correlations[score_file] = correlations

	save_object = {'original_activations':original_activations,
				   'circuit_activations':circuit_activations,
				   'correlations':correlations}

	torch.save(save_object,out_folder+score_file)

	print(time()-start)

torch.save(all_correlations,'resnet18_all_correlations.pt')