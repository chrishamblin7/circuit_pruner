#this file presume ranks and target accuracies for the feature have already been calculated 
# it returns a dictionary with meta data and the correlations of different masking levels given cumulative sparsity



import torch
from circuit_pruner.force import *
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


def mask_from_sparsity(rank_list, k):

	all_scores = torch.cat([torch.flatten(x) for x in rank_list])
	norm_factor = torch.sum(all_scores)
	all_scores.div_(norm_factor)

	threshold, _ = torch.topk(all_scores, k, sorted=True)
	acceptable_score = threshold[-1]
	cum_sal = torch.sum(threshold)

	mask = []

	for g in rank_list:
		mask.append(((g / norm_factor) >= acceptable_score).float())
		
	return mask,cum_sal

def mask_from_cum_salience(rank_list, cum_sal):

	all_scores = torch.cat([torch.flatten(x) for x in rank_list])
	norm_factor = torch.sum(all_scores)
	all_scores.div_(norm_factor)

	
	all_scores_sorted = torch.sort(all_scores, descending=True).values
	
	cum_total = 0.
	for i in range(len(all_scores_sorted)):
		cum_total += all_scores_sorted[i]
		if cum_total > cum_sal:
			print(i)
			threshold = all_scores_sorted[i]
			print(threshold)
			break
			

	mask = []

	for g in rank_list:
		mask.append(((g / norm_factor) >= threshold).float())
		
	return mask,i




import argparse
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--layer", type = str, 
						help='use circuit_pruner.force.show_model_layer_names(model) to see options')
	parser.add_argument("--config", type = str,default = 'configs/alexnet_sparse_config.py',help='relative_path to config file')
	parser.add_argument("--data-path", type = str, default = None, help='path to image data folder')
	parser.add_argument('--device', type = str, default='cuda:0', help='default "cuda:0"')  
	parser.add_argument('--method', type = str, default='actxgrad', help='default actxgrad') 
	parser.add_argument('--structure', type = str, default='edges', help='default edges') 
	parser.add_argument('--unit', type=int,help='numeral for unit in layer of target feature')
	parser.add_argument('--batch-size', type=int, default=200,
						help='number of batches for dataloader')             
	parser.add_argument('--by-sparsity', action='store_true', default=False, help='get masks based on sparsity rather than cumulative salience')




	args = parser.parse_args()
	return args


if __name__ == '__main__':

	args = get_args()
	print(args)

	by_sparsity = args.by_sparsity

	#which saliencies to check (EDITABLE)

	if not by_sparsity:
		cum_sals = [.99,.95]
		for i in range(18):
			cum_sals.append(round(cum_sals[-1]-.05,2))
		sparsities = []
	else:
		sparsities = [.5,.2,.1,.05,.01,.005,.001]
		cum_sals = []





	start = time.time()

	layer = args.layer
	unit = args.unit

	feature_name = layer+':'+str(unit)
	method = args.method
	structure = args.structure
	device = args.device
	batch_size = args.batch_size

	

	#params

	config = args.config

	if '/' in config:
		config_root_path = ('/').join(config.split('/')[:-1])
		update_sys_path(config_root_path)
	config_module = config.split('/')[-1].replace('.py','')
	params = __import__(config_module)

	data_path = params.data_path
	imageset = data_path.split('/')[-1]

	#get ranks
	ranks_folder = 'circuit_ranks/'+params.name+'/'+imageset+'/'+method+'/'

	for f in os.listdir(ranks_folder):
		if feature_name in f:
			layer_ranks = torch.load(ranks_folder+f)
			break



	if method == 'actxgrad':	
		rank_list = []

		for l in range(len(layer_ranks['ranks'][structure][method])):
			print(layer_ranks['ranks'][structure][method][l][0])
			rank_list.append(torch.tensor(layer_ranks['ranks'][structure][method][l][1]))
		
	elif method == 'snip':
		for l in layer_ranks['ranks']:
			l = l.to('cpu')
		layer_ranks = layer_ranks['ranks']

		if structure == 'weights':
			rank_list = layer_ranks
		elif structure in ['kernels','edges']:
			rank_list = []
			for grad in layer_ranks:
				if len(grad.shape) == 4: #conv2d layer
					rank_list.append(torch.mean(grad,dim = (2,3))) #average across height and width of each kernel
		else:
			rank_list = []
			for grad in layer_ranks:
				if len(grad.shape) == 4: #conv2d layer
					rank_list.append(torch.mean(grad,dim = (1,2,3))) #average across channel height and width of each filter
			

	



	#target_activations
	for act_file in os.listdir('target_activations/'+params.name+'/'+imageset+'/'):
		target_activations = torch.load('target_activations/'+params.name+'/'+imageset+'/'+act_file) 
		try:
			if feature_name in target_activations['activations'].keys():
				target_activations = target_activations['activations'][feature_name]
				break
		except:
			target_activations = torch.load('target_activations/'+params.name+'/'+imageset+'/'+feature_name+'.pt')[feature_name]

		

		
	#model

	model = params.model

	feature_target = {layer:[unit]}


	pruned_model = deepcopy(model)
	pruned_model = pruned_model.to(device)

	setup_net_for_circuit_prune(pruned_model, feature_targets=feature_target, rank_field = 'image',save_target_activations=True)

	pruned_model = pruned_model.to(device)

	reset_masks_in_net(pruned_model)


	#dataloader
	kwargs = {'num_workers': params.num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}
	dataloader = data.DataLoader(rank_image_data(params.data_path,
												params.preprocess,
												label_file_path = params.label_file_path,class_folders=True),
												batch_size=batch_size,
												shuffle=False,
												**kwargs)



	#total params


	total_params = 0
	for l in pruned_model.modules():
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



	#set up cum sals



		
	pearsons = []

	if not by_sparsity:

		for cum_sal in cum_sals:

			#setup mask
			mask,k = mask_from_cum_salience(rank_list, cum_sal)
			sparsities.append(float(k)/total_params)


			if structure is not 'weights':
				expanded_mask = expand_structured_mask(mask,pruned_model) #this weight mask will get applied to the network on the next iteration
			else:
				expanded_mask = mask

			for l in expanded_mask:
				l = l.to(device)

			#import pdb; pdb.set_trace()
			###GET ACTIVATIONS FROM PRUNED MODEL
			#get feature outputs from pruned model
			if structure == 'filters':
				reset_masks_in_net(pruned_model)
				apply_filter_mask(pruned_model,mask) #different than masking weights, because it also masks biases
			else:
				apply_mask(pruned_model,expanded_mask) 
				
				
			#run model
			save_target_activations_in_net(pruned_model,save=True)

			iter_dataloader = iter(dataloader)
			iters = len(iter_dataloader)



			pruned_target_activations = {}

			for it in range(iters):
				#clear_feature_targets_from_net(pruned_model)

				# Grab a single batch from the training dataset
				inputs, targets = next(iter_dataloader)
				inputs = inputs.to(device)

				pruned_model.zero_grad()

				#Run model forward until all targets reached
				try:
					outputs = pruned_model.forward(inputs)
				except:
					#except:
					pass


				activations = get_saved_target_activations_from_net(pruned_model)
				for l in activations:
					activations[l] = activations[l].to('cpu')
					if l not in pruned_target_activations.keys():
						pruned_target_activations[l] = activations[l]
					else:
						pruned_target_activations[l] = torch.cat((pruned_target_activations[l],activations[l]),dim=0)

			#compare
			cor = pearsonr(target_activations.flatten().numpy(),pruned_target_activations[feature_name].flatten().numpy())[0]
			if cor == np.nan:
				cor = 0.
			print(cor)
			
			pearsons.append(cor)

	else:

		for sparse in sparsities:

			#setup mask
			k = ceil(total_params*sparse)
			mask,cum_sal = mask_from_sparsity(rank_list,k)

			cum_sals.append(cum_sal)


			if structure is not 'weights':
				expanded_mask = expand_structured_mask(mask,pruned_model) #this weight mask will get applied to the network on the next iteration
			else:
				expanded_mask = mask

			for l in expanded_mask:
				l = l.to(device)

			#import pdb; pdb.set_trace()
			###GET ACTIVATIONS FROM PRUNED MODEL
			#get feature outputs from pruned model
			if structure == 'filters':
				reset_masks_in_net(pruned_model)
				apply_filter_mask(pruned_model,mask) #different than masking weights, because it also masks biases
			else:
				apply_mask(pruned_model,expanded_mask) 
				
				
			#run model
			save_target_activations_in_net(pruned_model,save=True)

			iter_dataloader = iter(dataloader)
			iters = len(iter_dataloader)



			pruned_target_activations = {}

			for it in range(iters):
				#clear_feature_targets_from_net(pruned_model)

				# Grab a single batch from the training dataset
				inputs, targets = next(iter_dataloader)
				inputs = inputs.to(device)

				pruned_model.zero_grad()

				#Run model forward until all targets reached
				try:
					outputs = pruned_model.forward(inputs)
				except:
					#except:
					pass


				activations = get_saved_target_activations_from_net(pruned_model)
				for l in activations:
					activations[l] = activations[l].to('cpu')
					if l not in pruned_target_activations.keys():
						pruned_target_activations[l] = activations[l]
					else:
						pruned_target_activations[l] = torch.cat((pruned_target_activations[l],activations[l]),dim=0)

			#compare
			cor = pearsonr(target_activations.flatten().numpy(),pruned_target_activations[feature_name].flatten().numpy())[0]
			if cor == np.nan:
				cor = 0.
			print(cor)
			
			pearsons.append(cor)



	save_object = {
		'correlations':pearsons,
		'keep_ratios':sparsities,
		'cum_sals':cum_sals,
		'method':method,
		'layer':layer,
		'unit':unit,
		'structure':structure,
		'batch_size':batch_size,
		'data_path':data_path,
		'config':args.config,
		'by_sparsity':by_sparsity
			}

	
	save_folder = 'cum_salience_accuracies/'+params.name+'/'+imageset+'/'+method
	if not os.path.exists(save_folder):
		os.makedirs(save_folder,exist_ok=True)
	torch.save(save_object,save_folder+'/%s_%s_%s.pt'%(params.name,feature_name,str(time.time())))


	print(time.time() - start)
