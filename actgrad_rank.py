#given a feature this returns scores, no mask



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


	imageset = data_path.split('/')[-1]


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



	save_object = {'ranks':layer_ranks,
				'layer':layer,
				'unit':unit,
				'rank_field':rank_field,
				'structure':structure,
				'batch_size':batch_size,
				'data_path':data_path,
				'config':args.config
					}

	if not os.path.exists('circuit_ranks/'+params.name+'/'+imageset+'/actgrad'):
		os.makedirs('circuit_ranks/'+params.name+'/'+imageset+'/actgrad',exist_ok=True)
	torch.save(save_object,'circuit_ranks/%s/actgrad/%s_%s:%s_%s.pt'%(params.name,params.name,layer,str(unit),str(time.time())))


	print(time.time() - start)
