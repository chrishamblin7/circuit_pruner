#given a feature this returns scores, no mask

import torch
from circuit_pruner.force import *
from circuit_pruner.extraction import *
from circuit_pruner.custom_exceptions import TargetReached
import time
import os
from circuit_pruner.utils import update_sys_path
import torch.utils.data as data
import torchvision.datasets as datasets
from circuit_pruner.data_loading import rank_image_data
from circuit_pruner.dissected_Conv2d import *
from copy import deepcopy
import numpy as np
from scipy.stats import spearmanr, pearsonr
from math import ceil




'''
class ModelBreak(Exception):
	"""Base class for other exceptions"""
	pass

class TargetReached(ModelBreak):
	"""Raised when the output target for a subgraph is reached, so the model doesnt neeed to be run forward any farther"""
	pass   
'''


##DATA LOADER###
import torch.utils.data as data
import torchvision.datasets as datasets
from circuit_pruner.data_loading import rank_image_data
from circuit_pruner.dissected_Conv2d import *
from copy import deepcopy



import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--layer", type = str, 
						help='use circuit_pruner.force.show_model_layer_names(model) to see options')
	parser.add_argument("--config", type = str,default = 'configs/alexnet_sparse_config.py',
						help='relative_path to config file')
	parser.add_argument("--data-path", type = str, default = None, help='path to image data folder')
	parser.add_argument('--device', type = str, default='cuda:0', help='default "cuda:0"')
	parser.add_argument('--T', type = int, default=10, help='number of FORCE iterations,default 10')
	parser.add_argument('--structure', type = str, default='kernels', help='default kernels')
	parser.add_argument('--unit', type=int,help='numeral for unit in layer of target feature')
	parser.add_argument('--batch-size', type=int, default=200,
						help='number of batches for dataloader')            
	parser.add_argument('--sparsity', type=float, default=.5,
						help='number of batches for dataloader')
	parser.add_argument('--dont-save-masks', action='store_true', default=False, help='dont save the masks in the output dictionary')
	parser.add_argument('--rank-field', type=str, default='image',
						help='what activations in the target feature do we care about (image, min, or max)')             
	parser.add_argument('--get_accuracies', action='store_true', default=False, 
						help='store activations for feature of original model and subgraph')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_args()
	print(args)

	#get variables from config
	if '/' in args.config:
		config_root_path = ('/').join(args.config.split('/')[:-1])
		update_sys_path(config_root_path)
	config_module = args.config.split('/')[-1].replace('.py','')
	params = __import__(config_module)


	model = params.model


	layer = args.layer
	unit = args.unit
	device= args.device
	rank_field = args.rank_field
	structure = args.structure
	method = 'FORCE'
	T = args.T
	sparsity = args.sparsity
	data_path = params.data_path
	imageset = data_path.split('/')[-1]
	feature_name = layer+':'+str(unit)

	if args.data_path is None:
		data_path = params.data_path
	else:
		data_path = args.data_path

	if args.batch_size is None:
		batch_size = params.batch_size
	else:
		batch_size = args.batch_size


	kwargs = {'num_workers': params.num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}


	label_file_path =  params.label_file_path


	dataloader = data.DataLoader(rank_image_data(data_path,
												params.preprocess,
												label_file_path = label_file_path,class_folders=True),
												batch_size=batch_size,
												shuffle=False,
												**kwargs)

	
	start = time.time()


	#GET RANKS 
	start = time.time()

	model = model.to(device)
	masked_model = deepcopy(model)


	feature_target = {layer:[unit]}


	#generate weight mask with FORCE pruner
	ranks,mask = circuit_FORCE_pruning(masked_model, dataloader, feature_targets = feature_target,feature_targets_coefficients = None, T=T,full_dataset = True, keep_ratio=sparsity, num_params_to_keep=None, device=device, structure=structure, rank_field = rank_field, mask=None, setup_net=True, return_ranks = True)    #progressive skeletonization

	#get structured mask
	if structure == 'weights':
		rank_list = ranks.to('cpu')
	elif structure in ['kernels','edges']:
		rank_list = []
		for grad in ranks:
			if len(grad.shape) == 4: #conv2d layer
				rank_list.append(torch.mean(grad,dim = (2,3)).to('cpu')) #average across height and width of each kernel
	else:
		rank_list = []
		for grad in ranks:
			if len(grad.shape) == 4: #conv2d layer
				rank_list.append(torch.mean(grad,dim = (1,2,3)).to('cpu')) #average across channel height and width of each filter


	#total params
	total_params = 0
	for l in masked_model.modules():
		if isinstance(l, nn.Conv2d):
			if not l.last_layer:  #all params potentially relevant
				if structure in ['kernels','edges']:
					total_params += int(l.weight.shape[0]*l.weight.shape[1])
				else:
					total_params += int(l.weight.shape[0])

			else: #only weights leading into feature targets are relevant
				if structure in ['kernels','edges']:
					total_params += int(len(l.feature_targets_indices)*l.weight.shape[1])
				else:
					total_params += len(l.feature_targets_indices)
				break


	#setup original mask
	print('target sparsity: %s'%str(sparsity))
	print('total params to feature: %s'%str(total_params))

	k = ceil(total_params*sparsity)

	print('kept params in original mask: %s'%str(k))
	
	orig_mask_sum = 0
	for l in mask:
		orig_mask_sum += int(torch.sum(l))




	#run model
	save_target_activations_in_net(masked_model,save=True)

	iter_dataloader = iter(dataloader)
	iters = len(iter_dataloader)


	#save the pre extraction target activations, we might want to know what they were later
	masked_target_activations = {}

	for it in range(iters):
		#clear_feature_targets_from_net(pruned_model)

		# Grab a single batch from the training dataset
		inputs, targets = next(iter_dataloader)
		inputs = inputs.to(device)

		masked_model.zero_grad()

		#Run model forward until all targets reached
		try:
			outputs = masked_model.forward(inputs)
		except:
			#except:
			pass


		activations = get_saved_target_activations_from_net(masked_model)
		for l in activations:
			activations[l] = activations[l].to('cpu')
			if l not in masked_target_activations.keys():
				masked_target_activations[l] = activations[l]
			else:
				masked_target_activations[l] = torch.cat((masked_target_activations[l],activations[l]),dim=0)



	print('masked target activations time: %s'%str(time.time()-start))


	#check model for 'collapse', after applying the mask, there may be kernels remaining in the model that no longer have any causal connection to
	# the target feature, all paths to the feature have been masked. we want to remove these edges as well, calculating a new 'effective sparsity'.

	start = time.time()

	effect_mask = kernel_mask_2_effective_kernel_mask(mask)
	

	#live inputs, the pruned model might not have inputs leading to all 3 input channels, so we need to check
	#for those so we can get rid of those channels of the input images

	live_input_channels = []

	for i in range(effect_mask[0][0].shape[0]):
		tot = torch.sum(effect_mask[0][:,i])
		if tot > 0:
			live_input_channels.append(i)

	#check for TOTAL COLLAPSE (there is no path to the target feature, the extracted circuit is literally nothing)
	total_collapse = False
	effective_sum  = 0
	for l in effect_mask:
		effective_sum += int(torch.sum(l))
	if effective_sum == 0:
		print('TOTAL COLLAPSE')
		total_collapse = True


	if not total_collapse:
		#extract model with the effective_mask,

		#del masked_model
		

		pruned_model = extract_circuit_with_eff_mask(model,effect_mask)

		#del model

		pruned_model = pruned_model.to(device)


		feature_target = {layer:[0]} #new model only has 1 output, and its the feature target

		print('prune model time: %s'%str(time.time()-start))

		#Get activations from pruned_model

		start = time.time()

		masked_pruned_model = deepcopy(pruned_model)    #were making a copy not to mask, but to 'setup', so we can use all the machinery like fetching target activations
		pruned_model = pruned_model.to('cpu')
		setup_net_for_circuit_prune(masked_pruned_model, feature_targets=feature_target, rank_field = 'image',save_target_activations=True)
		masked_pruned_model = masked_pruned_model.to(device)
		reset_masks_in_net(masked_pruned_model)

		save_target_activations_in_net(masked_pruned_model,save=True)

		iter_dataloader = iter(dataloader)
		iters = len(iter_dataloader)


		#save the pre extraction target activations, we might want to know what they were later
		pruned_target_activations = {}



		for it in range(iters):
			#clear_feature_targets_from_net(pruned_model)

			# Grab a single batch from the training dataset
			inputs, targets = next(iter_dataloader)
			inputs = inputs.to(device)
			inputs = inputs[:,live_input_channels]

			masked_pruned_model.zero_grad()

			#Run model forward until all targets reached
			try:
				outputs = masked_pruned_model.forward(inputs)
			except:
				#except:
				pass


			activations = get_saved_target_activations_from_net(masked_pruned_model)
			for l in activations:
				activations[l] = activations[l].to('cpu')
				if l not in pruned_target_activations.keys():
					pruned_target_activations[l] = activations[l]
				else:
					pruned_target_activations[l] = torch.cat((pruned_target_activations[l],activations[l]),dim=0)

		pruned_model

		print('pruned target activations time: %s'%str(time.time()-start))


	#save outputs
	save_object = {
		'sparsity':sparsity,
		'unit':unit,
		'layer':layer,
		'method':method,
		'layer':layer,
		'unit':unit,
		'structure':structure,
		'batch_size':batch_size,
		'data_path':data_path,
		'config':args.config,
		'masked_target_activations':masked_target_activations,
		'feature_name':feature_name,
		'total_collapse':total_collapse,
		'total_params':total_params,
		'masked_k':k,
		'effective_k':effective_sum,
		'effective_sparsity':float(effective_sum)/float(total_params),
		'device':device,
			}

	if not total_collapse:
		save_object['pruned_model'] = pruned_model
		save_object['pruned_target_activations'] = pruned_target_activations


	if not args.dont_save_masks:
		for i,l in enumerate(mask):
			mask[i] = mask[i].to('cpu')
		for l in effect_mask:
			effect_mask[i] = effect_mask[i].to('cpu')

		save_object['mask'] = mask
		save_object['effective_mask'] = effect_mask



	save_folder = 'extracted_circuits/'+params.name+'/'+imageset+'/'+method
	if not os.path.exists(save_folder):
		os.makedirs(save_folder,exist_ok=True)
	torch.save(save_object,save_folder+'/%s_%s_%s_%s.pt'%(params.name,feature_name,str(sparsity),str(T)))


	print(time.time() - start)
