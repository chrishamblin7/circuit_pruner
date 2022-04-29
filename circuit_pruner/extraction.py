from scipy.stats import spearmanr, pearsonr
import numpy as np
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




def mask_2_effective_mask(mask):
	
	effective_mask = deepcopy(mask)
	for i in range(len(effective_mask)):
		effective_mask[i] = effective_mask[i].to('cpu')

	#pixel to feature connectivity check
	prev_filter_mask = None
	for i in range(len(effective_mask)):
		if prev_filter_mask is not None:  # if we arent in first layer, we have to eliminate kernels connecting to 'dead' filters from the previous layer
			if len(effective_mask[i].shape) == 4:
				effective_mask[i] = (prev_filter_mask*effective_mask[i].permute(0,2,3,1)).permute(0,3,1,2)
			else:
				effective_mask[i] = prev_filter_mask*effective_mask[i]

		#now we need to get the filter mask for this layer, (masking those filters with no kernels leading in)
		prev_filter_mask = torch.zeros(effective_mask[i].shape[0])
		for j in range(effective_mask[i].shape[0]):
			if not torch.all(torch.eq(effective_mask[i][j],torch.zeros(effective_mask[i][0].shape))):
				prev_filter_mask[j] = 1

	#reverse direction feature to pixel connectivity check
	prev_filter_mask = None
	for i in reversed(range(len(effective_mask))):
		if prev_filter_mask is not None:  # if we arent in last layer, we have to eliminate kernels connecting to 'dead' filters from the previous layer
			if len(effective_mask[i].shape) == 4:
				effective_mask[i] = (prev_filter_mask*effective_mask[i].permute(1,2,3,0)).permute(3,0,1,2)
			else:
				effective_mask[i] = torch.transpose(prev_filter_mask*torch.transpose(effective_mask[i],0,1),0,1)

		#now we need to get the filter mask for this layer, (masking those filters with no kernels leading in)
		prev_filter_mask = torch.zeros(effective_mask[i].shape[1])
		for j in range(effective_mask[i].shape[1]):
			if not torch.all(torch.eq(effective_mask[i][:,j],torch.zeros(effective_mask[i][:,0].shape))):
				prev_filter_mask[j] = 1

	orig_sum  = 0
	for l in mask:
		orig_sum += int(torch.sum(l))
	print('original mask: %s params'%str(orig_sum))

	effective_sum  = 0
	for l in effective_mask:
		effective_sum += int(torch.sum(l))
	print('effective mask: %s params'%str(effective_sum))

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
					if len(layer_mask[i].shape) == 1: #linear or kernel mask
						if not torch.all(torch.eq(layer_mask[i],torch.zeros(layer_mask.shape[1]))):
							out_channels.append(i) 
					else: #conv weight mask
						if not torch.all(torch.eq(layer_mask[i],torch.zeros(layer_mask.shape[1],layer_mask.shape[2],layer_mask.shape[3]))):
							out_channels.append(i) 
						
				for i in range(layer_mask.shape[1]):
					if len(layer_mask[:,i].shape) == 1: #linear or kernel mask
						if not torch.all(torch.eq(layer_mask[:,i],torch.zeros(layer_mask.shape[0]))):
							in_channels.append(i)
					else: #conv weight mask
						if not torch.all(torch.eq(layer_mask[:,i],torch.zeros(layer_mask.shape[0],layer_mask.shape[2],layer_mask.shape[3]))):
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
						if len(layer_mask.shape) == 2:
							if layer_mask[o,i] != 0:
								weights[o_i,i_i,:,:] = old_conv.weight[o,i,:,:]
						else:
							weights[o_i,i_i,:,:] = old_conv.weight[o,i,:,:]*layer_mask[o,i,:,:]

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



def mask_from_sparsity(rank_list, k):

	all_scores = torch.cat([torch.flatten(x) for x in rank_list])
	norm_factor = torch.sum(abs(all_scores))
	all_scores.div_(norm_factor)

	all_scores = all_scores.type(torch.float)

	threshold, _ = torch.topk(all_scores, k, sorted=True)
	acceptable_score = threshold[-1]
	cum_sal = torch.sum(threshold)

	if acceptable_score == 0:
		print('gradients from this feature are sparse,\
the minimum acceptable rank at this sparsity has a score of zero! \
we will return a mask thats smaller than you asked, by masking all \
parameters with a score of zero.')


	mask = []

	for g in rank_list:
		mask.append(((g / norm_factor) > acceptable_score).float())
		
	return mask,cum_sal




def dissected_model_ranks_2_circuit_model(layer_ranks,sparsity,model,feature_targets,device,method='actxgrad',structure='edges',use_effective_mask=True,rank_field='image'):
	
	rank_list = []

	if isinstance(layer_ranks,list):
		rank_list = layer_ranks

	else:
		if 'ranks' in layer_ranks.keys():
			layer_ranks = layer_ranks['ranks']

		for l in range(len(layer_ranks[structure][method])):
			if len(layer_ranks[structure][method][l][1].nonzero()[1])>0:
				rank_list.append(torch.tensor(layer_ranks[structure][method][l][1]).to('cpu'))

	#model
	model.to('cpu').eval()
	masked_model = deepcopy(model)
	masked_model = masked_model.to(device).eval()


	setup_net_for_circuit_prune(masked_model,feature_targets, rank_field = rank_field)



	reset_masks_in_net(masked_model)
	
	
	#total params
	total_params = 0
	num_zero_params = 0
	for l in masked_model.modules():
		if isinstance(l, nn.Conv2d):
			if not l.last_layer:  #all params potentially relevant
				if structure in ['kernels','edges']:
					total_params += int(l.weight.shape[0]*l.weight.shape[1])
					kernel_sums = torch.sum(torch.abs(l.weight), (2,3), keepdim=False)
					num_zero_params += int((kernel_sums == 0).sum())
					
				elif structure in ['filters','nodes']:
					total_params += int(l.weight.shape[0])
				else:
					total_params += int(l.weight.shape[0]*l.weight.shape[1])*int(l.weight.shape[2]*l.weight.shape[3])
					num_zero_params += int((l.weight == 0).sum())

			else: #only weights leading into feature targets are relevant
				if structure in ['kernels','edges']:
					total_params += int(len(l.feature_targets_indices)*l.weight.shape[1])
					target_params = torch.index_select(l.weight, 0, torch.tensor(l.feature_targets_indices).to(device))
					kernel_sums = torch.sum(torch.abs(target_params), (2,3), keepdim=False)
					num_zero_params += int((kernel_sums == 0).sum())
				elif structure in ['filters','nodes']:
					total_params += len(l.feature_targets_indices)
				else: #each individual weight is a removable param
					total_params += int(len(l.feature_targets_indices)*l.weight.shape[1])*int(l.weight.shape[2]*l.weight.shape[3])
					target_params = torch.index_select(l.weight, 0, torch.tensor(l.feature_targets_indices).to(device))
					num_zero_params += int((target_params == 0).sum())
				break
	
	#setup original mask
	print('target sparsity: %s'%str(sparsity))
	print('total params to feature: %s\n'%str(total_params))
	if num_zero_params != 0:
		print('we found %s params that were already zero\'d out, your model is already a pruned circuit right? . . . \
			   just making sure, we are subtracting these params from the total.\n'%str(num_zero_params))
		total_params = total_params-num_zero_params
		print('new total params: %s    (after subtracting previously masked params)'%str(total_params))

		

	k = ceil(total_params*sparsity)

	print('kept params in original mask: %s      (total params * sparsity)'%str(k))
	#generate mask
	mask,cum_sal = mask_from_sparsity(rank_list,k) 
	


	orig_mask_sum = 0
	for l in mask:
		orig_mask_sum += int(torch.sum(l))
	print(orig_mask_sum)

	if structure is not 'weights':
		expanded_mask = expand_structured_mask(mask,masked_model) #this weight mask will get applied to the network on the next iteration
	else:
		expanded_mask = mask

	for l in expanded_mask:
		l = l.to(device)


	#apply mask
	if structure == 'filters':
		reset_masks_in_net(masked_model)
		apply_filter_mask(masked_model,mask) #different than masking weights, because it also masks biases
	else:
		apply_mask(masked_model,expanded_mask) 

	if use_effective_mask:
		effect_mask = mask_2_effective_mask(mask)
		#effect_mask = kernel_mask_2_effective_kernel_mask(mask)

		#check for TOTAL COLLAPSE (there is no path to the target feature, the extracted circuit is literally nothing)

		total_collapse = False
		effective_sum  = 0
		for l in effect_mask:
			effective_sum += int(torch.sum(l))
		print('effective_sparsity: %s'%str(effective_sum/total_params))
		if effective_sum == 0:
			print('TOTAL COLLAPSE')
			total_collapse = True


		if not total_collapse:
			pruned_model = extract_circuit_with_eff_mask(model,effect_mask)


	if not total_collapse:
		return pruned_model,effect_mask
	else:
		return None,None










# def dissected_model_ranks_2_circuit_model(layer_ranks,sparsity,model,feature_targets,device,method='actxgrad',structure='edges',use_effective_mask=True,rank_field='image'):
	
# 	rank_list = []
	
# 	if isinstance(layer_ranks,list):
# 		rank_list = layer_ranks

# 	else:
# 		if 'ranks' in layer_ranks.keys():
# 			layer_ranks = layer_ranks['ranks']

# 		for l in range(len(layer_ranks[structure][method])):
# 			if len(layer_ranks[structure][method][l][1].nonzero()[1])>0:
# 				rank_list.append(torch.tensor(layer_ranks[structure][method][l][1]).to('cpu'))

# 	#model

# 	masked_model = deepcopy(model)
# 	masked_model = masked_model.to(device)


# 	setup_net_for_circuit_prune(masked_model,feature_targets, rank_field = rank_field)



# 	reset_masks_in_net(masked_model)

# 	#total params
# 	total_params = 0
# 	for l in masked_model.modules():
# 		if isinstance(l, nn.Conv2d):
# 			if not l.last_layer:  #all params potentially relevant
# 				if structure in ['kernels','edges']:
# 					total_params += int(l.weight.shape[0]*l.weight.shape[1])
# 				elif structure in ['filters','nodes']:
# 					total_params += int(l.weight.shape[0])
# 				else:
# 					total_params += int(l.weight.shape[0]*l.weight.shape[1])*int(l.weight.shape[2]*l.weight.shape[3])


# 			else: #only weights leading into feature targets are relevant
# 				if structure in ['kernels','edges']:
# 					total_params += int(len(l.feature_targets_indices)*l.weight.shape[1])
# 				elif structure in ['filters','nodes']:
# 					total_params += len(l.feature_targets_indices)
# 				else: #each individual weight is a removable param
# 					total_params += int(len(l.feature_targets_indices)*l.weight.shape[1])*int(l.weight.shape[2]*l.weight.shape[3])
# 				break


# 	#setup original mask
# 	print('target sparsity: %s'%str(sparsity))
# 	print('total params to feature: %s'%str(total_params))

# 	k = ceil(total_params*sparsity)

# 	print('kept params in original mask: %s'%str(k))
# 	#generate mask
# 	mask,cum_sal = mask_from_sparsity(rank_list,k)


# 	orig_mask_sum = 0
# 	for l in mask:
# 		orig_mask_sum += int(torch.sum(l))


# 	if structure is not 'weights':
# 		expanded_mask = expand_structured_mask(mask,masked_model) #this weight mask will get applied to the network on the next iteration
# 	else:
# 		expanded_mask = mask

# 	for l in expanded_mask:
# 		l = l.to(device)


# 	#apply mask
# 	if structure == 'filters':
# 		reset_masks_in_net(masked_model)
# 		apply_filter_mask(masked_model,mask) #different than masking weights, because it also masks biases
# 	else:
# 		apply_mask(masked_model,expanded_mask) 

# 	if use_effective_mask:
# 		effect_mask = kernel_mask_2_effective_kernel_mask(mask)

# 		#check for TOTAL COLLAPSE (there is no path to the target feature, the extracted circuit is literally nothing)
# 		total_collapse = False
# 		effective_sum  = 0
# 		for l in effect_mask:
# 			effective_sum += int(torch.sum(l))
# 		print('effective_sparsity: %s'%str(effective_sum/total_params))
# 		if effective_sum == 0:
# 			print('TOTAL COLLAPSE')
# 			total_collapse = True


# 		if not total_collapse:
# 			pruned_model = extract_circuit_with_eff_mask(model,effect_mask)


# 	if not total_collapse:
# 		return pruned_model,effect_mask

def get_preservation_at_sparsities(model,ranks,feature_targets,dataloader,sparsities,device,rank_field='image',metric='avg_diff'):

    scores = {}
    
    print('original')
    #get original model activations
    model.to('cpu').eval()
    orig_model = deepcopy(model)
    orig_model = orig_model.to(device).eval()
    setup_net_for_circuit_prune(orig_model, feature_targets=feature_targets,save_target_activations=True)
    
    #pass data
    iter_dataloader = iter(dataloader)
    iters = len(iter_dataloader)
    for it in range(iters):
        inputs, target = next(iter_dataloader)
        inputs = inputs.to(device)
        try:
            output = orig_model(inputs)
        except TargetReached:
            pass
        
    #fetch activations
    orig_acts_all = get_saved_target_activations_from_net(orig_model)

    


    orig_acts = {}
    for feature in orig_acts_all.keys():
        
        scores[feature] = []
        orig_acts_all[feature] = torch.from_numpy(orig_acts_all[feature])
        
        
        if rank_field == 'image':
            orig_acts[feature] = orig_acts_all[feature].mean(dim=(1, 2))
        elif rank_field in ['max','orig_max']:
            orig_acts[feature] = orig_acts_all[feature].view(orig_acts_all[feature].size(0), orig_acts_all[feature].size(1)*orig_acts_all[feature].size(2)).max(dim=-1).values
        elif rank_field == 'min':
            orig_acts[feature] = orig_acts_all[feature].view(orig_acts_all[feature].size(0), orig_acts_all[feature].size(1)*orig_acts_all[feature].size(2)).min(dim=-1).values
        elif isinstance(rank_field,list):
            orig_acts[feature] = []
            for i in range(len(rank_field)):
                orig_acts[feature].append(orig_acts_all[feature][i,int(rank_field[i][0]),int(rank_field[i][1])])
            orig_acts[feature] = torch.tensor(orig_acts[feature])
            
        orig_acts[feature] = orig_acts[feature].detach().cpu().numpy().astype('float32')

        
    #get max activation indices
    if rank_field == 'orig_max':
        rank_field = {}
        for feature in orig_acts_all.keys():
            rank_field[feature] = []
            for i in range(orig_acts_all[feature].shape[0]):
                rank_field[feature].append(np.unravel_index(orig_acts_all[feature][i].argmax(), orig_acts_all[feature][i].shape))
    
    print(orig_acts)
    print(rank_field)
   
    for sparsity in sparsities:
        print(sparsity)
        #extract circuit
        circuit, mask = dissected_model_ranks_2_circuit_model(ranks,sparsity,model,feature_targets,device)
        circuit.to(device).eval()
        
        #get activations from ciruit
        setup_net_for_circuit_prune(circuit, feature_targets=feature_targets,save_target_activations=True)
        
        #pass data
        iter_dataloader = iter(dataloader)
        iters = len(iter_dataloader)
        for it in range(iters):
            inputs, target = next(iter_dataloader)
            inputs = inputs.to(device)
            try:
                output = circuit(inputs)
            except TargetReached:
                pass
        
        #fetch activations
        acts_all = get_saved_target_activations_from_net(circuit)
        acts = {}
        for feature in acts_all.keys():
            
            acts_all[feature] = torch.from_numpy(acts_all[feature])
            
            if rank_field == 'image':
                acts[feature] = acts_all[feature].mean(dim=(1, 2))
            elif rank_field in ['max','orig_max']:
                acts[feature] = acts_all[feature].view(acts_all[feature].size(0), acts_all[feature].size(1)*acts_all[feature].size(2)).max(dim=-1).values
            elif rank_field == 'min':
                acts[feature] = acts_all[feature].view(acts_all[feature].size(0), acts_all[feature].size(1)*acts_all[feature].size(2)).min(dim=-1).values
            elif isinstance(rank_field,list):
                acts[feature] = []
                for i in range(len(rank_field)):
                    acts[feature].append(acts_all[feature][i,int(rank_field[i][0]),int(rank_field[i][1])])
                acts[feature] = torch.tensor(acts[feature])
            elif isinstance(rank_field,dict):
                acts[feature] = []
                for i in range(len(rank_field[feature])):
                    acts[feature].append(acts_all[feature][i,int(rank_field[feature][i][0]),int(rank_field[feature][i][1])])
                acts[feature] = torch.tensor(acts[feature])
                
            acts[feature] = acts[feature].detach().cpu().numpy().astype('float32')
            
            print(acts)
        
            #caculate score (how close is circuit activation to original?)
            if metric == 'spearman':
                score = spearmanr(orig_acts[feature],acts[feature]).correlation
                if score is np.nan:
                    score = 0.
            elif metric == 'pearson':
                score = pearsonr(orig_acts[feature],acts[feature])[0]
                if score is np.nan:
                    score = 0.
            elif metric == 'avg_diff':
                score =  np.mean(acts[feature]-orig_acts[feature])
            elif metric == 'normed_diff':
                norm_factor = np.std(orig_acts[feature])
                score = np.mean(acts[feature]-orig_acts[feature])/norm_factor
            
            print('SCORE: %s'%str(score))
            scores[feature].append(score)

    return scores

