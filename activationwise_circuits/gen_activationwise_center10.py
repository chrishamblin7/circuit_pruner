#well load general packages here, 
#API specific functions will be loaded right before they are used, so you can see where they come from

import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
import numpy as np
from torch import nn
import os
from circuit_pruner.data_loading import single_image_data
from circuit_pruner.simple_api.target import positional_loss
import time
import pickle
import pandas as pd
from circuit_pruner.simple_api.target import sum_abs_loss, positional_loss
from circuit_pruner.simple_api.target import feature_target_saver

overwrite = False

all_layers_num = {'features.3':range(50,192),
              'features.10':range(50,256),
              'features.6':range(50,384),
              'features.8':range(50,256)}

device = 'cuda:0'

root_out_folder = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/activationwise_circuits/center_10_sample/'

from circuit_pruner.simple_api.score import actgrad_filter_extractor

def actgrad_filter_score(model,dataloader,target_layer_name,unit,loss_f=sum_abs_loss,absolute=True,return_target=False):
    all_layers = OrderedDict([*model.named_modules()])
    scoring_layers = []
    for layer in all_layers:
        if layer == target_layer_name:   #HACK MIGHT NOT WORK WITH INCEPTION
            break
        if isinstance(all_layers[layer],torch.nn.modules.conv.Conv2d):
            scoring_layers.append(layer)
            
    _ = model.eval()
    device = next(model.parameters()).device 
    
    scores = OrderedDict()
    
    
    overall_loss = 0
    with feature_target_saver(model,target_layer_name,unit) as target_saver:
        with actgrad_filter_extractor(model,scoring_layers,absolute = absolute) as score_saver:
            for i, data in enumerate(dataloader, 0):
                inputs, label = data
                inputs = inputs.to(device)

                model.zero_grad() #very import!
                target_activations = target_saver(inputs)

                #feature collapse
                loss = loss_f(target_activations)
                overall_loss+=loss
                loss.backward()

            #get average by dividing result by length of dset
            activations = score_saver.activations
            gradients = score_saver.gradients

            for l in scoring_layers:
                layer_scores = (activations[l] * gradients[l]).mean(dim=(1,2))
                if l not in scores.keys():
                    scores[l] = layer_scores
                else:
                    scores[l] += layer_scores


    remove_keys = []
    for layer in scores:
        if torch.sum(scores[layer]) == 0.:
            remove_keys.append(layer)
    if len(remove_keys) > 0: 
        print('removing layers from scores with scores all 0:')
        for k in remove_keys:
            print(k)
            del scores[k]


    model.zero_grad() 
    if return_target:
        return scores,float(overall_loss.detach().cpu())
    else:
        return scores

import numpy as np
import umap


def umap_scores(act_scores,layer='all',norm_data=False):
    data = []
    acts = []
    norms = []
    
    for sample in act_scores:
        if layer == 'all':
            v = []
            norm = 0
            for l in sample['scores']:
                v = v+list(sample['scores'][l].flatten())
                norm+= float(sample['scores'][l].sum())
        else:
            v = list(sample['scores'][layer].flatten())
            norm = sample['scores'][layer].sum()
        
        
        if norm_data:
            v = list(torch.tensor(v)/torch.norm(torch.tensor(v)))
        data.append(v)
        acts.append(float(sample['activation']))
        norms.append(norm)
    
    data = np.array(data)
    mapper = umap.UMAP().fit(data)
    
    out_data = mapper.fit_transform(data)
    return out_data,acts,norms
                




from torchvision.models import resnet18, alexnet

#model = resnet18(pretrained=True)
model = alexnet(pretrained=True)
_ = model.to(device).eval()

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from circuit_pruner.utils import load_config
from circuit_pruner.data_loading import rank_image_data


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

unnormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])


preprocess =  transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                normalize])

kwargs = {'num_workers': 4, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}



config_file = '../configs/alexnet_sparse_config.py'

config = load_config(config_file)

# dataloader = DataLoader(rank_image_data(config.data_path,
#                                         config.preprocess,
#                                         label_file_path = config.label_file_path,class_folders=True),
#                                         batch_size=64,
#                                         shuffle=False,
#                                         **kwargs)

data_folder = '/mnt/data/datasets/imagenet/train10/'

batch_dataloader = DataLoader(rank_image_data(data_folder,
                                        config.preprocess,
                                        class_folders=False,
                                        return_image_name=True),
                                        batch_size=64,
                                        shuffle=False,
                                        **kwargs)


#images = os.listdir(data_folder)
#images.sort()


# from circuit_pruner.receptive_fields import *

# input_size = (3,224,224)
# rf_dict = receptive_field(model.features, input_size)

# layerwise_center_positions = {'features.0':(rf_dict['1']['output_shape'][2]//2,rf_dict['1']['output_shape'][3]//2),
#                               'features.3':(rf_dict['4']['output_shape'][2]//2,rf_dict['4']['output_shape'][3]//2),
#                               'features.6':(rf_dict['7']['output_shape'][2]//2,rf_dict['7']['output_shape'][3]//2),
#                               'features.8':(rf_dict['9']['output_shape'][2]//2,rf_dict['9']['output_shape'][3]//2),
#                               'features.10':(rf_dict['11']['output_shape'][2]//2,rf_dict['11']['output_shape'][3]//2),        
#                              }

layerwise_center_positions = {'features.0': (27, 27),
                              'features.3': (13, 13),
                              'features.6': (6, 6),
                              'features.8': (6, 6),
                              'features.10': (6, 6)}

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - float(value))).argmin()
    if value >= array[idx]:
        return idx
    else:
        return idx-1
    
from circuit_pruner.simple_api.target import layer_saver

all_layers = ['features.3','features.6','features.8','features.10']


#get activations

image_names = []

from circuit_pruner.simple_api.target import layer_saver

all_layer_activations = {}
for l in all_layers:
    all_layer_activations[l] = []

for i, data in enumerate(batch_dataloader):
    if i%10==0:
        print(i)
    if i==400:
        break
    images = data[0].to(device)
    image_names += data[2]
    with layer_saver(model, all_layers) as extractor:
        batch_layer_activations = extractor(images) #all features for layer and all images in batch
        for l in all_layers:
            all_layer_activations[l].append(batch_layer_activations[l][:,:,layerwise_center_positions[l][0],layerwise_center_positions[l][1]])
        
for l in all_layers:
    all_layer_activations[l] = torch.cat(all_layer_activations[l])


from circuit_pruner.receptive_fields import *

def recep_field_crop(image,model,layer,target_position,rf_dict = None):
    """
    inputs: a tensor image, model, layer name, and position in the layers activation map (H,W)
    outputs: cropped image at receptive field for that image
    """
    if rf_dict is None:
        input_size = tuple(image.shape)
        rf_dict = receptive_field(model.features, input_size)
    
    pos = receptive_field_for_unit(rf_dict, layer, target_position)
    return image[:,int(pos[0][0]):int(pos[0][1]),int(pos[1][0]):int(pos[1][1])]

#save cropped images
from PIL import Image

input_size = (3,224,224)
rf_dict = receptive_field(model.features, input_size)   

load_image =   transforms.Compose([
                                    transforms.Resize((input_size[1],input_size[2])),
                                    transforms.ToTensor()])
topil = transforms.ToPILImage()
# print('saving cropped images')
# for i,image_name in enumerate(image_names):
#     for layer in all_layers:
#         viz_layer_name = layer.replace('.','_')
#         image_path = data_folder+image_name
#         position = layerwise_center_positions[layer]
#         image = Image.open(image_path)
#         tensor_image = load_image(image)
#         cropped_tensor_image = recep_field_crop(tensor_image,model,viz_layer_name,position,rf_dict = rf_dict)
#         img = topil(cropped_tensor_image)
#         output_name = '.'.join(image_name.split('.')[:-1])+'pos_%s_%s'%(position[0],position[1])+'.'+image_name.split('.')[-1]
#         img.save(root_out_folder+'/images/'+layer+'/'+output_name)




for score_layer_name in all_layers:
    print(score_layer_name)
    viz_layer_name = score_layer_name.replace('.','_')
    for unit in all_layers_num[score_layer_name]:
        print(unit)
        start = time.time()

        unit_activations = all_layer_activations[score_layer_name][:,unit]


        #lets get a minimum, maximum and center activation from each image
        data = []

        #ew
        for i in range(unit_activations.shape[0]):

            #center activation
            center_pos = layerwise_center_positions[score_layer_name]
            data.append({'image':image_names[i],
                        'activation':float(unit_activations[i].detach().cpu()),
                        'layer':score_layer_name,
                        'unit':unit,
                        'position':(center_pos[0],center_pos[1])})

        #scoring
        for i,d in enumerate(data):
            #if i%400==0:
            #    print(i)
            image_path = os.path.join(data_folder,d['image'])
            dataloader = DataLoader(single_image_data(image_path,
                                                        config.preprocess),
                                    batch_size=1,
                                    shuffle=False
                                    )
            
            position = d['position']
            #skip if position at center
            #if center_data[i]['position'] == position:
            #    continue
            
            pos_loss = positional_loss(position)
            
            scores = actgrad_filter_score(model,dataloader,score_layer_name,unit,loss_f=pos_loss)
            data[i]['scores'] = scores

        #save score
        outname = '%s_%s_scores.pkl'%(viz_layer_name,unit)
        #import pdb;pdb.set_trace()
        pickle.dump(data,open(os.path.join(root_out_folder,'scores',outname),'wb'))


        #save umap df
        data_map,acts,norms = umap_scores(data, layer = 'all',norm_data=False)
        data_map_normed,acts,norms = umap_scores(data, layer = 'all',norm_data=True)


        columns = ['x','y','x_normed','y_normed','image','position','activation','norm']

        big_list = []
        for i in range(len(data)):
            position = data[i]['position']
            image_name = '.'.join(data[i]['image'].split('.')[:-1])+'pos_%s_%s'%(position[0],position[1])+'.'+data[i]['image'].split('.')[-1]
            x = data_map[i][0]
            y = data_map[i][1]
            x_normed = data_map_normed[i][0]
            y_normed = data_map_normed[i][1]
            activation = acts[i]
            norm = norms[i]
            big_list.append([x,y,x_normed,y_normed,image_name,position,activation,norm])
            

        df = pd.DataFrame(big_list,columns=columns)
        outname_df = '%s_%s_umap_df.pkl'%(viz_layer_name,unit)
        pickle.dump(df,open(root_out_folder+'/umap_dfs/'+outname_df,'wb'))
        #pickle.dump(sample_data,open(output_folder+'scores.pkl','wb'))

        
        print(time.time()-start)

