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
from math import log, exp, ceil
from circuit_pruner.simple_api.target import feature_target_saver,sum_abs_loss
from circuit_pruner.simple_api.mask import mask_from_scores, setup_net_for_mask, apply_mask
from circuit_pruner.simple_api.util import params_2_target_from_scores
from circuit_pruner.simple_api.dissected_Conv2d import *

####  SCORES ####

'''
Functions for computing saliency scores for parameters on models
'''

def snip_score(model,dataloader,target_layer_name,unit,layer_types_2_score = [nn.Conv2d,nn.Linear],loss_f = sum_abs_loss):

	_ = model.eval()
	device = next(model.parameters()).device  
	layers = OrderedDict([*model.named_modules()])

	#target_saver = feature_target_saver(model,layer_name,unit)
	scores = OrderedDict()
	with feature_target_saver(model,target_layer_name,unit) as target_saver:
		for i, data in enumerate(dataloader, 0):

			inputs, label = data
			inputs = inputs.to(device)
			#label = label.to(device)

			model.zero_grad() #very import!
			target_activations = target_saver(inputs)

			#feature collapse
			loss = loss_f(target_activations)
			loss.backward()

			#get weight-wise scores
			for layer_name,layer in layers.items():
				if type(layer) not in layer_types_2_score:
					continue
					
				if layer.weight.grad is None:
					continue

				try: #does the model have a weight mask?
					#scale scores by batch size (*inputs.shape)
					layer_scores = torch.abs(layer.weight_mask.grad).detach().cpu()*inputs.shape[0]

				except:
					layer_scores = torch.abs(layer.weight*layer.weight.grad).detach().cpu()*inputs.shape[0]

				
				if layer_name not in scores.keys():
					scores[layer_name] = layer_scores
				else:
					scores[layer_name] += layer_scores
					
	# # eliminate layers with stored but all zero scores
	remove_keys = []
	for layer in scores:
		if torch.sum(scores[layer]) == 0.:
			remove_keys.append(layer)
	if len(remove_keys) > 0: 
		print('removing layers from scores with scores all 0:')
		for k in remove_keys:
			print(k)
			del scores[k]
		  
	#target_saver.hook.remove() # this is important or we will accumulate hooks in our model
	model.zero_grad() 

	return scores


def force_score(model, dataloader,target_layer_name,unit,keep_ratio=.1, T=10, num_params_to_keep=None, structure='weights',layer_types_2_score = [nn.Conv2d,nn.Linear],loss_f = sum_abs_loss, apply_final_mask = True, min_max=False):    #progressive skeletonization
	'''
	TO DO: This does not currently work with structured pruning, when target
	is a linear layer.
	'''

	assert structure in ('weights','kernels','filters')

	device = next(model.parameters()).device  
	

	_ = model.eval()

	
	setup_net_for_mask(model)
	layers = OrderedDict([*model.named_modules()])


	#before getting the schedule of sparsities well get the total
	#parameters into the target by running the scoring function once

	scores = snip_score(model,dataloader,target_layer_name,unit,layer_types_2_score = layer_types_2_score, loss_f = loss_f)
	if structure in ['kernels','filters']:
		structured_scores = structure_scores(scores, model, structure=structure)
	else:
		structured_scores = scores

	if min_max:
		structured_scores = minmax_norm_scores(structured_scores)

	#total params
	# total_params = 0
	# for layer_name, layer_scores in structured_scores.items():
	# 	if layer_name == target_layer_name:
	# 		#EDIT, this might not be general in cases like branching models
	# 		#only weights leading into feature targets are important
	# 		total_params += params_2_target_in_layer(unit,layers[layer_name])
	# 	else:
	# 		total_params += torch.numel(layer_scores)

	total_params = params_2_target_from_scores(structured_scores,unit,target_layer_name,model)
	
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

	
	#iteratively apply mask and score
	print("Pruning with %s pruning steps"%str(T))
	for t in range(1,T+1):
		
		print('step %s'%str(t))
		
		k = ceil(exp(t/T*log(num_params_to_keep)+(1-t/T)*log(total_params))) #exponential schedulr
		 
		print('%s params'%str(k))

		#mask model
		mask = mask_from_scores(structured_scores,num_params_to_keep=k)
		apply_mask(model,mask,zero_absent=False)

		#SNIP
		scores = snip_score(model,dataloader,target_layer_name,unit,layer_types_2_score = layer_types_2_score, loss_f = loss_f)
		if structure in ['kernels','filters']:
			structured_scores = structure_scores(scores, model, structure=structure)
		else:
			structured_scores = scores
		
		if min_max:
			structured_scores = minmax_norm_scores(structured_scores)

	#do we alter the final model to have the mask 
	# prescribed by FORCE, or keep it unmasked?
	if apply_final_mask:
		'applying final mask to model'
		mask = mask_from_scores(structured_scores,num_params_to_keep=k)
		apply_mask(model,mask,zero_absent=False)

		#print about final mask
		mask_ones = 0
		for layer_name,layer_mask in mask.items():
			mask_ones += int(torch.sum(layer_mask))
		print('final mask: %s/%s params (%s)'%(mask_ones,total_params,mask_ones/total_params))
	else:
		'keeping model unmasked'
		setup_net_for_mask(model) #sets mask to all 1s


	return structured_scores



def actgrad_kernel_score(model,dataloader,target_layer_name,unit,loss_f = sum_abs_loss):

	_ = model.eval()
	device = next(model.parameters()).device 

	dis_model = dissect_model(deepcopy(model))
	_ = dis_model.to(device).eval()

	model.to('cpu') #we need as much memory as we can get

	all_layers = OrderedDict([*dis_model.named_modules()])
	dissected_layers = OrderedDict()

	for layer_name, layer in all_layers.items():
		if isinstance(layer,dissected_Conv2d):
			dissected_layers[layer_name] = layer

	#target_saver = feature_target_saver(model,layer_name,unit)
	scores = OrderedDict()
	with feature_target_saver(dis_model,target_layer_name,unit) as target_saver:
		for i, data in enumerate(dataloader, 0):
			#print('batch: '+str(i))
			inputs, label = data
			inputs = inputs.to(device)
			#label = label.to(device)

			dis_model.zero_grad() #very import!
			target_activations = target_saver(inputs)

			#feature collapse
			loss = loss_f(target_activations)
			loss.backward()

			#get weight-wise scores
			for layer_name,layer in dissected_layers.items():

				if layer.kernel_scores is None:
					if layer_name in scores.keys():
						raise Exception('kernel scores for %s not stored for batch %s'%(layer_name,str(i)))
					else:
						continue


				layer_scores = layer.kernel_scores
				
				if layer_name not in scores.keys():
					scores[layer_name] = layer_scores
				else:
					scores[layer_name] += layer_scores

	#reshape scores to in-out dimensions
	flattened_scores = OrderedDict()
	for layer_name, score in scores.items():
		flattened_scores[layer_name] = dissected_layers[layer_name].unflatten_kernel_scores( scores = scores[layer_name])

	del scores
	scores = flattened_scores


					
	# # eliminate layers with stored but all zero scores
	remove_keys = []
	for layer in scores:
		if torch.sum(scores[layer]) == 0.:
			remove_keys.append(layer)
	if len(remove_keys) > 0: 
		print('removing layers from scores with scores all 0:')
		for k in remove_keys:
			print(k)
			del scores[k]


	for layer_name, layer in dissected_layers.items():
		layer.kernel_scores = None
	del dis_model #might be redundant

	return scores



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
		
	structured_scores = OrderedDict()
	for layer_name in scores:
		if isinstance(layers[layer_name],nn.Conv2d):
			structured_scores[layer_name] = torch.mean(scores[layer_name],dim=collapse_dims)
			
	return structured_scores      
		

def minmax_norm_scores(scores, min=0, max=1):
	out_scores = OrderedDict()
	for layer_name, scores in scores.items():
		old_min = torch.min(scores)
		old_max = torch.max(scores)
		out_scores[layer_name] = (scores - old_min)/(old_max - old_min)*(max - min) + min

	return out_scores





