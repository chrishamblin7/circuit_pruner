#functions for 'extracting' a circuit from a masked model to a new neural network

import torch
from copy import deepcopy
from circuit_pruner.force import *

#check model for 'collapse', after applying the mask, there may be kernels remaining in the model that no longer have any causal connection to
# the target feature, all paths to the feature have been masked. we want to remove these edges as well, calculating a new 'effective sparsity'.

def kernel_mask_2_effective_kernel_mask(kernel_mask):
	effective_mask = deepcopy(kernel_mask)
	for i in range(len(effective_mask)):
		effective_mask[i] = effective_mask[i].to('cpu')

	#pixel to feature connectivity check
	prev_filter_mask = None
	for i in range(len(effective_mask)):
		if prev_filter_mask is not None:  # if we arent in first layer, we have to eliminate kernels connecting to 'dead' filters from the previous layer
			effective_mask[i] = prev_filter_mask*effective_mask[i]

		#now we need to get the filter mask for this layer, (masking those filters with no kernels leading in)
		prev_filter_mask = torch.zeros(effective_mask[i].shape[0])
		for j in range(effective_mask[i].shape[0]):
			if not torch.all(torch.eq(effective_mask[i][j],torch.zeros(effective_mask[i].shape[1]))):
				prev_filter_mask[j] = 1

	#reverse direction feature to pixel connectivity check
	prev_filter_mask = None
	for i in reversed(range(len(effective_mask))):
		if prev_filter_mask is not None:  # if we arent in last layer, we have to eliminate kernels connecting to 'dead' filters from the previous layer
			effective_mask[i] = torch.transpose(prev_filter_mask*torch.transpose(effective_mask[i],0,1),0,1)

		#now we need to get the filter mask for this layer, (masking those filters with no kernels leading in)
		prev_filter_mask = torch.zeros(effective_mask[i].shape[1])
		for j in range(effective_mask[i].shape[1]):
			if not torch.all(torch.eq(effective_mask[i][:,j],torch.zeros(effective_mask[i].shape[0]))):
				prev_filter_mask[j] = 1

	orig_sum  = 0
	for l in kernel_mask:
		orig_sum += int(torch.sum(l))
	print('original mask: %s kernels'%str(orig_sum))

	effective_sum  = 0
	for l in effective_mask:
		effective_sum += int(torch.sum(l))
	print('effective mask: %s kernels'%str(effective_sum))

	return effective_mask






def extract_circuit_with_eff_mask(model,eff_mask):
	#this is currently hacky only works on models with all nn.sequential or .features module
	model.to('cpu')

	#hack
	layer_names = show_model_layer_names(model,printer=False)
	constrained_layer_names = []
	for name in layer_names:
		if 'features_' in name:
			constrained_layer_names.append(name)


	
	for layer in model.children():
		if not isinstance(layer, nn.Conv2d):
			model = model.features
			break
		break
	 

	subgraph_model = nn.Sequential()


	
	l = 0 #layer index
	lc = 0  #conv layer index
	with torch.no_grad():
		for layer in model.children():
			if not isinstance(layer, nn.Conv2d):
				subgraph_model.add_module(constrained_layer_names[l], layer)
			else:
				old_conv = layer
				layer_mask = eff_mask[lc]
				out_channels = []
				in_channels = []
				for i in range(layer_mask.shape[0]):
					if not torch.all(torch.eq(layer_mask[i],torch.zeros(layer_mask.shape[1]))):
						out_channels.append(i) 
				for i in range(layer_mask.shape[1]):
					if not torch.all(torch.eq(layer_mask[:,i],torch.zeros(layer_mask.shape[0]))):
						in_channels.append(i)
				#initialize new conv with less filters
				new_conv = \
						torch.nn.Conv2d(in_channels = len(in_channels), \
						out_channels = len(out_channels) ,
						kernel_size = old_conv.kernel_size, \
						stride = old_conv.stride,
						padding = old_conv.padding,
						dilation = old_conv.dilation,
						groups = old_conv.groups,
						bias = (old_conv.bias is not None))     
				#reset weights       
				weights = new_conv.weight
				weights.fill_(0.)              
				#fill in unmasked weights from old conv

				for o_i,o in enumerate(out_channels):
					for i_i,i in enumerate(in_channels):
						weights[o_i,i_i,:,:] = old_conv.weight[o,i,:,:]

				#GENERATE BIAS 
				if new_conv.bias is not None:
					for o_i,o in enumerate(out_channels):
						new_conv.bias[o_i] = old_conv.bias[o]

				subgraph_model.add_module(constrained_layer_names[l], new_conv)

				lc+=1
				if lc == len(eff_mask):   # we are at the target layer
					break     
			l+=1


	return subgraph_model




