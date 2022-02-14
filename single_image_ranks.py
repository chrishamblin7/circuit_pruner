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

image_root = '/mnt/data/datasets/imagenet/all_images/'


feature_basis_images = pickle.load(open('../cnn_subgraph_visualizer/prepped_models/alexnet_sparse_temp/misc_data/top_bottom_images.pkl','rb'))


from time import time
import pickle

nodeids = range(1051,70,-13)
for nodeid in nodeids:
    nodeid -= 1

    layer_num,unit,layer_full_name = nodeid_2_perlayerid(nodeid,params)
    feature_name = layer_full_name+':'+str(unit)
    feature_target = {layer_full_name:[unit]}

    dissected_model = set_model_target_node(dissected_model,layer_full_name,unit)



    start = time()

    ranks_slow = {nodeid:{}}

    print(nodeid)
    print(feature_name)
    print(feature_to_conv_dict[layer_full_name])
    for side in ['max','min']:
        print('\nprocessing %s images\n'%side)
        for i, image_name in enumerate(feature_basis_images[feature_to_conv_dict[layer_full_name]][side]['image_names']):
            print(image_name)
            image_loader = data.DataLoader(single_image_data(image_root+image_name,
                                                    config.preprocess,
                                                    ),
                                                    batch_size=1,
                                                    shuffle=False,
                                                    **kwargs)

            iter_dataloader = iter(image_loader)
            inputs, targets = next(iter_dataloader)
            inputs = inputs.to(device)
            targets = targets.to(device)




            dissected_model.zero_grad()
            set_across_model(dissected_model,'rank_field',[feature_basis_images[feature_to_conv_dict[layer_full_name]][side]['positions'][i]])

            #Run model forward until all targets reached
            try:
                outputs = dissected_model(inputs)
            except:
                pass

            position1 = int(feature_basis_images[feature_to_conv_dict[layer_full_name]][side]['positions'][i][0])
            position2 = int(feature_basis_images[feature_to_conv_dict[layer_full_name]][side]['positions'][i][1])

            ranks_slow[nodeid][image_name+'_'+str(position1)+'_'+str(position2)] = {}
            ranks_slow[nodeid][image_name+'_'+str(position1)+'_'+str(position2)]['ranks'] = get_ranks_from_dissected_Conv2d_modules(dissected_model)
            ranks_slow[nodeid][image_name+'_'+str(position1)+'_'+str(position2)]['activation'] =  dissected_model.features[int(layer_full_name.split('_')[-1])].postbias_out[0,unit,position1,position2].cpu().detach().numpy().astype('float16')



    output_folder = '/mnt/data/chris/dropbox/Research-Hamblin/Projects/circuit_pruner_cvpr2022/circuit_ranks/alexnet_sparse/single_image_ranks/actxgrad/'
    pickle.dump(ranks_slow,open(output_folder+'node'+str(nodeid)+'_'+str(round(time(),2))+'.pt','wb'))
    print(time()-start)


