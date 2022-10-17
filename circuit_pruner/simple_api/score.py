import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, Iterable, Callable
from collections import OrderedDict
from circuit_pruner.custom_exceptions import TargetReached
import types
from copy import deepcopy
from circuit_pruner.simple_api.target import feature_target_saver,sum_abs_loss
from circuit_pruner.simple_api.mask import mask_from_scores, setup_net_for_mask, apply_mask

####  SCORES ####

'''
Functions for computing saliency scores for parameters on models
'''

def snip_score(model,dataloader,layer_name,unit,layer_types_2_score = [nn.Conv2d,nn.Linear],loss_f = sum_abs_loss):

	device = next(model.parameters()).device  
	layers = OrderedDict([*model.named_modules()])

	target_saver = feature_target_saver(model,layer_name,unit)
	scores = OrderedDict()  
	for i, data in enumerate(dataloader, 0):

		inputs, label = data
		inputs = inputs.to(device)
		#label = label.to(device)

		model.zero_grad() #very import!
		try:
			output = model(inputs)
		except TargetReached:
			pass

		#import pdb; pdb.set_trace()

		#feature collapse
		loss = loss_f(target_saver.target)
		loss.backward()

		#print(str(float(loss))+',')
		#if float(loss) == 7.885214328765869:
		#	import pdb; pdb.set_trace()

		#get weight-wise scores
		for layer_name,layer in layers.items():
			if type(layer) not in layer_types_2_score:
				continue
				
			if layer.weight.grad is None:
				continue
			
			if layer_name not in scores.keys():
				scores[layer_name] = torch.abs(layer.weight*layer.weight.grad).detach().cpu()/inputs.shape[0]
			#else:
			#	scores[layer_name] += torch.abs(layer.weight*layer.weight.grad).detach().cpu()/inputs.shape[0]
				
	# # eliminate layers with stored but all zero scores
	#print('layers eliminated with zero score')
	remove_keys = []
	for layer in scores:
		if torch.sum(scores[layer]) == 0.:
			print(layer)
			remove_keys.append(layer)
	for k in remove_keys:
		del scores[k]
		  
	target_saver.hook.remove() # this is important or we will accumulate hooks in our model
	model.zero_grad() 

	return scores


def force_score(model, dataloader, feature_targets = None,feature_targets_coefficients = None,keep_ratio=.1, T=10, num_params_to_keep=None, structure='weights'):    #progressive skeletonization
	
	assert structure in ('weights','kernels','filters')

	device = next(model.parameters()).device  
	

	_ = model.to(device).eval()

	net = net.to(device).eval()	
	for param in net.parameters():
		param.requires_grad = False
	
	if setup_net:
		setup_net_for_circuit_prune(net, feature_targets=feature_targets, score_field = score_field)
	
	
	#get total params given feature target might exclude some of network
	total_params = 0

	for layer in net.modules():
		if structure == 'weights' and (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
			if not layer.last_layer:  #all params potentially important
				total_params += len(layer.weight.flatten())
			else:    #only weights leading into feature targets are important
				total_params += len(layer.feature_targets_indices)*int(layer.weight.shape[1])
				break
		elif isinstance(layer, nn.Conv2d):
			if not layer.last_layer:  #all params potentially important
				if structure == 'kernels':
					total_params += int(layer.weight.shape[0]*layer.weight.shape[1])
				else:
					total_params += int(layer.weight.shape[0])
					
			else: #only weights leading into feature targets are important
				if structure == 'kernels':
					total_params += int(len(layer.feature_targets_indices)*layer.weight.shape[1])
				else:
					total_params += len(layer.feature_targets_indices)
				
				break
	
	if num_params_to_keep is None:
		num_params_to_keep = ceil(keep_ratio*total_params)
	else:
		keep_ratio = num_params_to_keep/total_params       #num_params_to_keep arg overrides keep_ratio
	
	print('pruning %s'%structure)
	print('total parameters: %s'%str(total_params))
	print('parameters after pruning: %s'%str(num_params_to_keep))
	print('keep ratio: %s'%str(keep_ratio))
  
	if num_params_to_keep >= total_params:
		print('num params to keep > total params, no pruning to do')
		return

	print("Pruning with %s pruning steps"%str(T))
	for t in range(1,T+1):
		
		print('step %s'%str(t))
		
		k = ceil(exp(t/T*log(num_params_to_keep)+(1-t/T)*log(total_params))) #exponential schedulr
		 
		print('%s params'%str(k))
		
		#SNIP
		if not return_scores:
			struct_mask = circuit_SNIP(net, dataloader, num_params_to_keep=k, feature_targets = feature_targets, feature_targets_coefficients = feature_targets_coefficients, use_abs_scores = use_abs_scores, structure=structure, mask=mask, full_dataset = full_dataset, device=device,setup_net=False)
		else:
			grads,struct_mask = circuit_SNIP(net, dataloader, num_params_to_keep=k, feature_targets = feature_targets, feature_targets_coefficients = feature_targets_coefficients, use_abs_scores = use_abs_scores, structure=structure, mask=mask, full_dataset = full_dataset, device=device,setup_net=False,return_scores=True)
		if structure is not 'weights':
			mask = expand_structured_mask(struct_mask,net) #this weight mask will get applied to the network on the next iteration
		else:
			mask = struct_mask

	apply_mask(net,mask)

	mask_total = 0
	mask_ones = 0
	for l in mask:
		mask_ones += int(torch.sum(l))
		mask_total += int(torch.numel(l))
	print('final mask: %s/%s params (%s)'%(mask_ones,mask_total,mask_ones/mask_total))


	if not return_scores:
		return struct_mask
	else:
		return grads,struct_mask


#### Score manipulations #####
'''
functions for manipulating scores
'''

def structure_scores(scores, model, structure='kernels'):
	
	assert structure in ['kernels','filters']
	layers = OrderedDict([*model.named_modules()])
	
	if structure == 'kernels':
		collapse_dims = (2,3)
	else:
		collapse_dims = (1,2,3)
		
	structured_scores = {}
	for layer_name in scores:
		if isinstance(layers[layer_name],nn.Conv2d):
			structured_scores[layer_name] = torch.mean(scores[layer_name],dim=collapse_dims)
			
	return structured_scores      
		

def minmax_norm_scores(scores, min=0, max=1):
	out_scores = {}
	for layer_name, scores in scores.items():
		old_min = torch.min(scores)
		old_max = torch.max(scores)
		out_scores[layer_name] = (scores - old_min)/(old_max - old_min)*(max - min) + min

	return out_scores



