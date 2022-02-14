#general
import torch
import os
from copy import deepcopy


#Pruner
from circuit_pruner.force import *
from circuit_pruner.utils import load_config


#dataloader
import torch.utils.data as data
import torchvision.datasets as datasets
from circuit_pruner.data_loading import rank_image_data, single_image_data
from circuit_pruner.dissected_Conv2d import *

from torchvision import transforms

import torch
import os
from copy import deepcopy
import pickle
import sys

#Pruner
from circuit_pruner.force import *
from circuit_pruner.utils import *
from circuit_pruner.receptive_fields import *


#dataloader
import torch.utils.data as data
import torchvision.datasets as datasets
from circuit_pruner.data_loading import rank_image_data, single_image_data
from circuit_pruner.dissected_Conv2d import *



device = 'cuda:2'

prepped_model_folder = 'alexnet_sparse_temp'

sys.path.insert(0,'../cnn_subgraph_visualizer/prepped_models/%s'%prepped_model_folder)

full_prepped_model_folder = '../cnn_subgraph_visualizer/prepped_models/%s'%prepped_model_folder

import prep_model_params_used as prep_model_params

params = {}
params['prepped_model'] = prepped_model_folder
params['prepped_model_path'] = full_prepped_model_folder
params['device'] = 'cuda:2'
params['deepviz_neuron'] = True
params['deepviz_edge'] = False


#Parameters

#Non-GUI parameters

#deepviz
params['deepviz_param'] = None
params['deepviz_optim'] = None
params['deepviz_transforms'] = None
params['deepviz_image_size'] = prep_model_params.deepviz_image_size

#backend
params['input_image_directory'] = prep_model_params.input_img_path+'/'   #path to directory of imput images you want fed through the network
params['preprocess'] = prep_model_params.preprocess     #torchvision transfrom to pass input images through
params['label_file_path'] = prep_model_params.label_file_path
params['criterion'] = prep_model_params.criterion
params['rank_img_path'] = prep_model_params.rank_img_path
params['num_workers'] = prep_model_params.num_workers
params['seed'] = prep_model_params.seed
params['batch_size'] = prep_model_params.batch_size
#params['dynamic_act_cache_num'] = 4  #max number of input image activations 'dynamic_activations' will have simultaneously


#load misc graph data
print('loading misc graph data')
misc_data = pickle.load(open('../cnn_subgraph_visualizer/prepped_models/%s/misc_graph_data.pkl'%prepped_model_folder,'rb'))
params['layer_nodes'] = misc_data['layer_nodes']
params['num_layers'] = misc_data['num_layers']
params['num_nodes'] = misc_data['num_nodes']
params['categories'] = misc_data['categories']
params['num_img_chan'] = misc_data['num_img_chan']
params['imgnode_positions'] = misc_data['imgnode_positions']
params['imgnode_colors'] = misc_data['imgnode_colors']
params['imgnode_names'] = misc_data['imgnode_names']
params['prepped_model_path'] = full_prepped_model_folder
params['ranks_data_path'] = full_prepped_model_folder+'/ranks/'



#config, load a configuration file

config = load_config('/mnt/data/chris/dropbox/Research-Hamblin/Projects/circuit_pruner_cvpr2022/configs/alexnet_sparse_config.py')


#make single image data loaders

kwargs = {'num_workers': config.num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}


feature_to_conv_dict = {'features_0':'conv0',
						'features_3':'conv1',
						'features_6':'conv2',
						'features_8':'conv3',
						'features_10':'conv4',}




#model
model = prep_model_params.model
model = model.eval().to(device)
dissected_model = dissect_model(deepcopy(model), store_ranks = True, clear_ranks = True, device=device)
dissected_model = dissected_model.to(device)


kwargs = {'num_workers': config.num_workers, 'pin_memory': True, 'sampler' : None} if 'cuda' in device else {}

image_root = '/mnt/data/chris/dropbox/Research-Hamblin/Projects/circuit_pruner_cvpr2022/image_data/bonw_circles/'

receptive_fields = None
if os.path.exists('../cnn_subgraph_visualizer/prepped_models/%s/receptive_fields.pkl'%prepped_model_folder):
	receptive_fields = pickle.load(open('../cnn_subgraph_visualizer/prepped_models/%s/receptive_fields.pkl'%prepped_model_folder,'rb'))
	 

from time import time
import pickle

#circle like images
nodeids = [693,657,803,399,445,446,500]
for nodeid in nodeids:

	layer_num,unit,layer_full_name = nodeid_2_perlayerid(nodeid,params)
	feature_name = layer_full_name+':'+str(unit)
	feature_target = {layer_full_name:[unit]}

	recep_field = receptive_field_for_unit(receptive_fields, layer_full_name, (5,5))


	dissected_model = set_model_target_node(dissected_model,layer_full_name,unit)
	set_across_model(dissected_model,'absolute_rank',False)


	start = time()

	ranks_slow = {nodeid:{}}

	print(nodeid)
	print(feature_name)
	print(feature_to_conv_dict[layer_full_name])

	image_names = os.listdir(image_root)
	image_names.sort()

	for i, image_name in enumerate(image_names):
		#if int(image_name.replace('.jpg','')) < 100:
		#	continue
		print(image_name)
		image_path = image_root+image_name
		image = Image.open(image_path)

		resize_2_tensor = transforms.Compose([transforms.Resize((int(recep_field[0][1]-recep_field[0][0]),int(recep_field[1][1]-recep_field[1][0]))),transforms.ToTensor()])
		tensor_image = resize_2_tensor(image)
		in_tensor = torch.zeros(3,params['deepviz_image_size'],params['deepviz_image_size'])
		in_tensor[:,int(recep_field[0][0]):int(recep_field[0][1]),int(recep_field[1][0]):int(recep_field[1][1])] = tensor_image

		norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

		in_tensor = norm(in_tensor)
		in_tensor = torch.unsqueeze(in_tensor,0)
		in_tensor = in_tensor.to(device)

		dissected_model.zero_grad()
		set_across_model(dissected_model,'rank_field',[[5,5]])

		#Run model forward until all targets reached
		try:
			outputs = dissected_model(in_tensor)
		except:
			pass

		position1 = 5
		position2 = 5

		ranks_slow[nodeid][image_name] = {}
		ranks_slow[nodeid][image_name]['ranks'] = get_ranks_from_dissected_Conv2d_modules(dissected_model)
		ranks_slow[nodeid][image_name]['activation'] =  dissected_model.features[int(layer_full_name.split('_')[-1])].postbias_out[0,unit,5,5].cpu().detach().numpy().astype('float16')



	output_folder = '/mnt/data/chris/dropbox/Research-Hamblin/Projects/circuit_pruner_cvpr2022/circuit_ranks/alexnet_sparse/bonw_circle_scale_ranks/actxgrad/'
	pickle.dump(ranks_slow,open(output_folder+'node'+str(nodeid)+'_noabsoluterank_'+str(round(time(),2))+'.pt','wb'))
	print(time()-start)


