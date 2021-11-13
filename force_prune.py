from circuit_pruner.force import *
import time
import os
from circuit_pruner.utils import update_sys_path


class ModelBreak(Exception):
	"""Base class for other exceptions"""
	pass

class TargetReached(ModelBreak):
	"""Raised when the output target for a subgraph is reached, so the model doesnt neeed to be run forward any farther"""
	pass   



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
	parser.add_argument("--category", type = str, default = None)
	parser.add_argument("--config", type = str,default = 'configs/alexnet_sparse_config.py',help='relative_path to config file')
	parser.add_argument("--data-path", type = str, default = None, help='path to image data folder')
	parser.add_argument('--device', type = str, default='cuda:0', help='default "cuda:0"')  
	parser.add_argument('--unit', type=int,help='numeral for unit in layer of target feature')
	parser.add_argument('--batch-size', type=int, default=None,
						help='number of batches for dataloader')
	parser.add_argument('--T', type=int, default=1,
						help='number of FORCE pruning iterations')
	parser.add_argument('--ratio', type=float, default=.05,
						help='ratio of params before and after pruning')
	parser.add_argument('--rank-field', type=str, default='image',
						help='what activations in the target feature do we care about (image, min, or max)')              
	parser.add_argument('--get_accuracies', action='store_true', default=False, help='store activations for feature of original model and subgraph')
	parser.add_argument('--structure', type=str, default='kernels', help='filters, kernels, or weights')



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
	T = args.T
	ratio = args.ratio
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


	image_loader = data.DataLoader(rank_image_data(data_path,
												params.preprocess,
												label_file_path = label_file_path,class_folders=True),
												batch_size=batch_size,
												shuffle=False,
												**kwargs)

	
	start = time.time()

	model = model.to(device)
	pruned_model = deepcopy(model)


	feature_target = {layer:[unit]}


	#generate weight mask with FORCE pruner
	mask = circuit_FORCE_pruning(pruned_model, image_loader, feature_targets = feature_target,
								T=T,full_dataset = True, keep_ratio=ratio, num_params_to_keep=None, 
								device=device, structure=structure, rank_field= rank_field, mask=None)

	###GET ACTIVATIONS
	#get feature outputs from pruned model 


	save_target_activations_in_net(pruned_model,save=True)

	iter_dataloader = iter(image_loader)
	iters = len(iter_dataloader)

	pruned_target_activations = {}

	if structure == 'filters':
		reset_masks_in_net(pruned_model)
		apply_filter_mask(pruned_model,mask) #different than masking weights, because it also masks biases


	for it in range(iters):
		#clear_feature_targets_from_net(pruned_model)

		# Grab a single batch from the training dataset
		inputs, targets = next(iter_dataloader)
		inputs = inputs.to(device)

		pruned_model.zero_grad()

		#Run model forward until all targets reached
		try:
			outputs = pruned_model.forward(inputs)
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

	#and original model
	#reset prune mask
	reset_masks_in_net(pruned_model)

	orig_target_activations = {}


	iter_dataloader = iter(image_loader)
	iters = len(iter_dataloader)


	for it in range(iters):
		clear_feature_targets_from_net(pruned_model)

		# Grab a single batch from the training dataset
		inputs, targets = next(iter_dataloader)
		inputs = inputs.to(device)

		pruned_model.zero_grad()

		#Run model forward until all targets reached
		try:
			outputs = pruned_model.forward(inputs)
		#except TargetReached:
		except:
			pass


		activations = get_saved_target_activations_from_net(pruned_model)
		for l in activations:
			activations[l] = activations[l].to('cpu')
			if l not in orig_target_activations.keys():
				orig_target_activations[l] = activations[l]
			else:
				orig_target_activations[l] = torch.cat((orig_target_activations[l],activations[l]),dim=0)


	del pruned_model
	torch.cuda.empty_cache()

	#save everything
	for i in range(len(mask)):
		mask[i] = mask[i].to('cpu')
	save_object = {'mask':mask,
				'full_target_activations':orig_target_activations,
				'pruned_target_activations':pruned_target_activations,
				'keep_ratio':ratio,
				'method':'FORCE',
				'T':T,
				'layer':layer,
				'unit':unit,
				'rank_field':rank_field,
				'structure':structure,
				'batch_size':batch_size,
				'data_path':data_path,
				'config':args.config
					}

	if not os.path.exists('circuit_masks/'+params.name+'/force'):
		os.mkdir('circuit_masks/'+params.name+'/force')
	torch.save(save_object,'circuit_masks/%s/force/%s_%s_unit%s_FORCE_%s_%s_%s.pt'%(params.name,params.name,layer,str(unit),str(ratio),str(T),str(time.time())))


	print(time.time() - start)




