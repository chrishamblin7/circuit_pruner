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


#### MASKS #####
'''
Functions for masking the network, given scores
'''

def mask_from_scores(scores,sparsity=None,num_params_to_keep=None):
	assert not ((sparsity is None) and (num_params_to_keep is None))

	keep_masks = {}
	
	#flatten
	scores_flat = torch.cat([torch.flatten(x) for x in scores.values()])
	norm_factor = torch.sum(abs(scores_flat))
	scores_flat.div_(norm_factor)

	#num kept params
	if not num_params_to_keep is None:
		k = num_params_to_keep
	else:
		k = int(len(scores_flat) * sparsity)

	#get threshold score
	threshold, _ = torch.topk(scores_flat, k, sorted=True)
	acceptable_score = threshold[-1]


	
	if acceptable_score == 0:
		print('gradients from this feature are sparse,\
			   the minimum acceptable score at this sparsity has a score of zero! \
				we will return a mask thats smaller than you asked, by masking all \
				parameters with a score of zero.')

	for layer_name in scores:
		layer_scores = scores[layer_name]
		keep_masks[layer_name] = (layer_scores / norm_factor > acceptable_score).float()
	
	return keep_masks


def masked_conv2d_forward(self, x):

	#pass input through conv and weight mask

	x = F.conv2d(x, self.weight * self.weight_mask, self.bias,
					self.stride, self.padding, self.dilation, self.groups) 

	return x

def masked_linear_forward(self, x):

	x = F.linear(x, self.weight * self.weight_mask, self.bias)

	return x

def setup_net_for_mask(model):

	#same naming trick as before 
	layers = OrderedDict([*model.named_modules()])

	for layer in layers.values():
		if isinstance(layer, nn.Conv2d):
			layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
			layer.forward = types.MethodType(masked_conv2d_forward, layer)
		elif isinstance(layer, nn.Linear):
			layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
			layer.forward = types.MethodType(masked_linear_forward, layer)
			
def apply_mask(model,mask, zero_absent=True):

	layers = OrderedDict([*model.named_modules()])
	setup_net_for_mask(model)

	#mask may be structured, lets 'expand' it before applying it to the model
	expanded_mask = expand_structured_mask(mask,model)

	for layer_name in expanded_mask:
		layers[layer_name].weight_mask = nn.Parameter(expanded_mask[layer_name].to(layers[layer_name].weight.device))
	if zero_absent:
		#mask all layers not specified in the mask
		for layer_name in layers:
			if layer_name not in expanded_mask.keys():
				try:
					layers[layer_name].weight_mask = nn.Parameter(torch.zeros_like(layers[layer_name].weight))
				except:
					pass



def expand_structured_mask(mask,model):
	'''Structured mask might have shape (filter, channel) for kernel structured mask, but the weights have
		shape (filter,channel,height,width), so we make a new weight wise mask based on the structured mask'''

	layers = OrderedDict([*model.named_modules()])
	expanded_mask = OrderedDict()

	for layer_name, layer_mask in mask.items():
		w = layers[layer_name].weight
		m = deepcopy(layer_mask)
		while len(m.shape) < len(w.shape):
			m = m.unsqueeze(dim=-1)
		m = m.expand(w.shape)
		expanded_mask[layer_name] = m
	
	return expanded_mask



def mask_intersect_over_union(mask1,mask2):
	iou = {}
	for layer_name in mask1:
		intersect_mask = mask1[layer_name]*mask2[layer_name]
		union_mask = torch.ceil((mask1[layer_name]+mask2[layer_name])/2)
		iou[layer_name] = (torch.sum(intersect_mask)/torch.sum(union_mask))
	return iou

