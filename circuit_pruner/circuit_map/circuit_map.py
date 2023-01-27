
from circuit_pruner.data_loading import single_image_data, rank_image_data, default_preprocess
from circuit_pruner.simple_api.target import sum_abs_loss, positional_loss, layer_activations_from_dataloader
from circuit_pruner.simple_api.score import actgrad_filter_score, actgrad_filter_extractor, get_num_params_from_cum_score
import umap
import torch
from torch import nn
import numpy as np
import os
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict

def umap_from_scores(scores,layers='all',norm_data=False,n_components=2):
  if layers=='all':
    layers = list(scores[0].keys())
  if isinstance(layers,str):
    layers = [layers]

  data = []
  l1_norms = []
  l2_norms = []
  #import pdb;pdb.set_trace()
  for sample in scores:
    traj_v = []
    for l in layers:
      traj_v = traj_v +list(sample[l].flatten())

    l1_norms.append(float(torch.tensor(traj_v).sum()))
    l2_norms.append(float(torch.norm(torch.tensor(traj_v))))

    if norm_data:
      traj_v = list(nn.functional.normalize(torch.tensor(traj_v),dim=0))
    data.append(traj_v) 

  data = np.array(data)
  mapper = umap.UMAP(n_components=n_components).fit(data)
    
  out_data = mapper.fit_transform(data)
  return out_data,l1_norms,l2_norms



def gen_image_trajectory_map_df(data_folder,model,target_layer,unit,
                                scores=None,preprocess=default_preprocess,
                                position=None,norm_data=False,umap_layer='all',
                                batch_size=64, target_layer_activations=None,n_components=2):
  '''
  required arguments:
    data_folder:  path to folder with just images with no sub-folders, oooor (not yet implemented) with class-wise subfolders
    model: A pytorch model
    target_layer: the name for the layer the trajectory map goes to; from "OrderedDict([*model.named_modules()]).keys()"
    unit: either an integer for a basis direction in the target_layer, or a list-like object of floats for a vector direction
  optional arguments:
    scores: scores per image (see simple_api.score), defaults to none nand computing these within the function
    preprocess: a torchvision transform, defaults to 224x224  resize and imagenet normalization
    position: only relevant for convolutional layers, which output an activation map, if position it specifies, it refers to a cell in this map, from which you backprop
    norm_data: do you want to normalize the trajectory vectors before running umap
    umap_layer: excepts a string layer name, like target layer, or a list of layer names. these are the layers whos scores are included in the trajectory vector. Defaults to "all" which includes all layers
  '''

  device = next(model.parameters()).device
  layers = OrderedDict([*model.named_modules()])
  all_images = os.listdir(data_folder)
  all_images.sort()

  if target_layer_activations is None:
    print('getting layer activations')
    target_layer_activations = layer_activations_from_dataloader(target_layer,data_folder,model,batch_size=batch_size)[target_layer]

  if isinstance(unit,int):
    unit_activations = target_layer_activations[:,unit]
  else:
    unit_activations = torch.tensordot(target_layer_activations, torch.tensor(unit).float(), dims=([1],[0]))


  assert not (len(unit_activations.shape)>1 and (position is None))
  # if len(unit_activations.shape>1) and (position is None):
  #   print('you did not specify a position but your layer returns multiple values per feature (it has an activation map). \n \
  #         Well average over this map, but consider specifying a position (as a tuple of ints).')

  if position is not None:
    for i in range(len(position)-1,-1,-1):
      unit_activations = unit_activations[..., position[i]]
  
  data = []
  for i in range(unit_activations.shape[0]):
    data.append({'image':all_images[i],
                'activation':float(unit_activations[i]),
                'layer':target_layer,
                'position':position
                })

  if scores is None:
    scores = []
    print('computing imagewise trajectory vectors')
    for i,d in enumerate(data):
        if i%100==0:
            print(str(i)+'/'+str(len(all_images)))
        image_path = os.path.join(data_folder,d['image'])
        dataloader = DataLoader(single_image_data(image_path,
                                                preprocess,
                                                rgb=True),
                                batch_size=1,
                                shuffle=False
                                )
        if position in d.keys():
          position = d['position']
          loss_func = positional_loss(position)
        else:
          loss_func = sum_abs_loss

        #use argument 'score_type == "activations" or score_type == "gradients"' to score with respect to those values per filter instead  
        image_scores = actgrad_filter_score(model,dataloader,target_layer,unit) 
        scores.append(image_scores)
      

    data_map,l1_norms,l2_norms = umap_from_scores(scores,norm_data=norm_data,layers=umap_layer,n_components=n_components) #umap of standarized image-wise trajectories through the network to the target feature
    #make a dataframe of umap data, this will make a consisent format that easier to save as a single object
    #we will load in some of these dataframes from google drive that I generated for each feature later on . . .

    columns = ['x','y','image','activation','l1_norm','l2_norm']
    if n_components == 3:
      columns.append('z')

    big_list = []
    for i in range(len(data)):
        image_name = data[i]['image']
        x = data_map[i][0]
        y = data_map[i][1]
        activation = float(unit_activations[i])
        l1_norm = l1_norms[i]
        l2_norm = l2_norms[i]
        row = [x,y,image_name,activation,l1_norm,l2_norm]
        if n_components == 3:
          row.append(data_map[i][2])
        big_list.append(row)
        
    umap_df = pd.DataFrame(big_list,columns=columns)

    return umap_df, scores
