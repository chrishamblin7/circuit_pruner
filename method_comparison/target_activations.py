from circuit_pruner.force import *
import time
import os
from circuit_pruner.utils import update_sys_path


##DATA LOADER###
import torch.utils.data as data
import torchvision.datasets as datasets
from circuit_pruner.data_loading import rank_image_data
from circuit_pruner.dissected_Conv2d import *
from copy import deepcopy



if __name__ == '__main__':



	config = '../configs/alexnet_sparse_config.py'
	layers = ['features_6','features_8','features_10']
	data_path = '../image_data/imagenet_2'
	units = range(20)
	device = 'cuda:0'
	batch_size = 200

	feature_targets = None
	#set this to select specific feature taer
	feature_targets = {
						'features_6':list(range(384)),
						'features_8':list(range(256)),
						'features_10':list(range(256))
						}

	#get variables from config
	if '/' in config:
		config_root_path = ('/').join(config.split('/')[:-1])
		update_sys_path(config_root_path)
	config_module = config.split('/')[-1].replace('.py','')
	params = __import__(config_module)


	model = params.model

	imageset = data_path.split('/')[-1]

	kwargs = {'num_workers': params.num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}


	label_file_path =  params.label_file_path


	image_loader = data.DataLoader(rank_image_data(data_path,
												params.preprocess,
												label_file_path = label_file_path,class_folders=True),
												batch_size=batch_size,
												shuffle=False,
												**kwargs)

	
	start = time.time()

	if feature_targets is None:
		feature_targets = {}

		for layer in layers:
			feature_targets[layer] = list(units)



	model = model.to(device)
	pruned_model = deepcopy(model)
	setup_net_for_circuit_prune(pruned_model, feature_targets=feature_targets, save_target_activations=True)
	save_target_activations_in_net(pruned_model,save=True)
	reset_masks_in_net(pruned_model)

	iter_dataloader = iter(image_loader)
	iters = len(iter_dataloader)



	#and original model
	#reset prune mask
	

	orig_target_activations = {}


	iter_dataloader = iter(image_loader)
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
		#except TargetReached:
		except:
			pass


		activations = get_saved_target_activations_from_net(pruned_model)

		for l in activations:
			activations[l] = torch.from_numpy(activations[l])
			#activations[l] = activations[l].to('cpu')
			if l not in orig_target_activations.keys():
				orig_target_activations[l] = activations[l]
			else:
				orig_target_activations[l] = torch.cat((orig_target_activations[l],activations[l]),dim=0)



	save_object = {'activations':orig_target_activations,
				'batch_size':batch_size,
				'data_path':data_path,
				'config':config
					}

	if not os.path.exists('./target_activations/'+params.name+'/'+imageset):
		os.makedirs('./target_activations/'+params.name+'/'+imageset,exist_ok=True)
	torch.save(save_object,'./target_activations/'+params.name+'/'+imageset+'/orig_activations.pt')

	print(time.time() - start)



