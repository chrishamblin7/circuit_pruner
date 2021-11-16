#given a feature a a ratio this returns a mask and meta data



from circuit_pruner.force import *
from circuit_pruner.custom_exceptions import TargetReached
import time
import os
from circuit_pruner.utils import update_sys_path

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
	parser.add_argument("--config", type = str,default = 'configs/alexnet_sparse_config.py',help='relative_path to config file')
	parser.add_argument("--data-path", type = str, default = None, help='path to image data folder')
	parser.add_argument('--device', type = str, default='cuda:0', help='default "cuda:0"')  
	parser.add_argument('--unit', type=int,help='numeral for unit in layer of target feature')
	parser.add_argument('--batch-size', type=int, default=7,
						help='number of batches for dataloader')
	parser.add_argument('--ratio', action='append', help='<Required> Set flag', required=True)
	parser.add_argument('--rank-field', type=str, default='image',
						help='what activations in the target feature do we care about (image, min, or max)')              
	parser.add_argument('--get_accuracies', action='store_true', default=False, help='store activations for feature of original model and subgraph')
	parser.add_argument('--structure', type=str, default='both', help='filters, kernels, or both')



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
	ratios = args.ratio
	device= args.device
	rank_field = args.rank_field
	structure = args.structure


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

	dissected_model = dissect_model(deepcopy(model), store_ranks = True, device=device)
	dissected_model = dissected_model.to(device)
	dissected_model = set_model_target_node(dissected_model,layer,unit)

	iter_dataloader = iter(dataloader)
	iters = len(iter_dataloader)

	for it in range(iters):
		# Grab a single batch from the training dataset
		inputs, targets = next(iter_dataloader)
		inputs = inputs.to(device)
		targets = targets.to(device)

		# Compute gradients (but don't apply them)
		dissected_model.zero_grad()

		#Run model forward until all targets reached
		try:
			outputs = dissected_model(inputs)
		except:
			pass
		
	#fetch ranks
	layer_ranks = get_ranks_from_dissected_Conv2d_modules(dissected_model)

	del dissected_model


	#RESET DATA LOADER WITH LARGER BATCH SIZE FOR UNDISSECTED MODEL
	dataloader = data.DataLoader(rank_image_data(data_path,
												params.preprocess,
												label_file_path = label_file_path,class_folders=True),
												batch_size=200,
												shuffle=False,
												**kwargs)




	#SETUP MASK MODEL
	feature_target = {layer:[unit]}
	

	pruned_model = deepcopy(model)
	pruned_model = pruned_model.to(device)

	setup_net_for_circuit_prune(pruned_model, feature_targets=feature_target, rank_field = rank_field,save_target_activations=True)

	pruned_model = pruned_model.to(device)

	#GET ORIGINAL ACTIVATIONS

	reset_masks_in_net(pruned_model)

	orig_target_activations = {}


	iter_dataloader = iter(dataloader)
	iters = len(iter_dataloader)


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
			if l not in orig_target_activations.keys():
				orig_target_activations[l] = activations[l]
			else:
				orig_target_activations[l] = torch.cat((orig_target_activations[l],activations[l]),dim=0)




	#GET MASKS

	if structure == 'both':
		structures = ['kernels','filters']
	else:
		structures = [structure]

	for ratio in ratios:
		import pdb;pdb.set_trace()
		ratio = float(ratio)

		for structure in structures:


			#get total params given feature target might exclude some of network
			total_params = 0

			for l in pruned_model.modules():
				if isinstance(l, nn.Conv2d):
					if not l.last_layer:  #all params potentially relevant
						if structure == 'kernels':
							total_params += int(l.weight.shape[0]*l.weight.shape[1])
						else:
							total_params += int(l.weight.shape[0])
							
					else: #only weights leading into feature targets are relevant
						if structure == 'kernels':
							total_params += int(len(l.feature_targets_indices)*l.weight.shape[1])
						else:
							total_params += len(l.feature_targets_indices)
						
						break

			num_params_to_keep = ceil(ratio*total_params)
			print(total_params)
			print(num_params_to_keep)

			
			for ranktype in ['act','actxgrad']:
				#generate weight mask from layer_ranks
				rank_list = []

				struct_dict = {'kernels':'edges','filters':'nodes'}

				for l in layer_ranks[struct_dict[structure]][ranktype]:
					rank_list.append(torch.tensor(l[1]))



				all_scores = torch.cat([torch.flatten(x) for x in rank_list])
				norm_factor = torch.sum(all_scores)
				all_scores.div_(norm_factor)

				threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
				acceptable_score = threshold[-1]

				mask = []

				

				for g in rank_list:
					mask.append(((g / norm_factor) >= acceptable_score).float())


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
						outputs = pruned_model(inputs)
					#except TargetReached:
					except:
						pass

					activations = get_saved_target_activations_from_net(pruned_model)
					for l in activations:
						activations[l] = activations[l].to('cpu')
						if l not in pruned_target_activations.keys():
							pruned_target_activations[l] = activations[l]
						else:
							pruned_target_activations[l] = torch.cat((pruned_target_activations[l],activations[l]),dim=0)




				#save everything
				for i in range(len(mask)):
					mask[i] = mask[i].to('cpu')
				save_object = {'mask':mask,
							'full_target_activations':orig_target_activations,
							'pruned_target_activations':pruned_target_activations,
							'keep_ratio':ratio,
							'method':ranktype,
							'layer':layer,
							'unit':unit,
							'rank_field':rank_field,
							'structure':structure,
							'batch_size':batch_size,
							'data_path':data_path,
							'config':args.config
								}

				if not os.path.exists('circuit_masks/'+params.name+'/actgrad'):
					os.mkdir('circuit_masks/'+params.name+'/actgrad')
				torch.save(save_object,'circuit_masks/%s/actgrad/%s_%s_unit%s_%s_%s_%s.pt'%(params.name,params.name,layer,str(unit),ranktype,str(ratio),str(time.time())))


	print(time.time() - start)
