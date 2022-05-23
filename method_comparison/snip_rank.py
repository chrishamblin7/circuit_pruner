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
	method = 'snip'
	data_path = params.data_path
	imageset = data_path.split('/')[-1]


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
	pruned_model = deepcopy(model)


	feature_target = {layer:[unit]}


	#generate weight mask with FORCE pruner
	ranks = circuit_snip_rank(pruned_model, dataloader, feature_targets = feature_target,
								full_dataset = True, 
								device=device, rank_field= rank_field, mask=None)



	save_object = {'ranks':ranks,
				'layer':layer,
				'unit':unit,
				'method':method,
				'rank_field':rank_field,
				'batch_size':batch_size,
				'data_path':data_path,
				'config':args.config
					}
	save_folder = 'circuit_ranks/'+params.name+'/'+imageset+'/'+method
	if not os.path.exists(save_folder):
		os.makedirs(save_folder,exist_ok=True)
	torch.save(save_object,save_folder+'/%s_%s:%s_%s.pt'%(params.name,layer,str(unit),str(time.time())))


	print(time.time() - start)
