device = 'cpu'

units = list(range(20))

layers = ['features_6','features_8','features_10']

config_path = '../configs/alexnet_sparse_config.py'
out_folder = './circuit_ranks/alexnet_sparse/imagenet_2/magnitude/'

import os
if not os.path.exists(out_folder):
    os.makedirs(out_folder,exist_ok=True)

import torch
from circuit_pruner.force import circuit_kernel_magnitude_ranking
from circuit_pruner.utils import load_config

config = load_config(config_path)
model = config.model.to(device)

for layer in layers:
    print(layer)
    for unit in units:
        print(unit)
        feature_targets = {layer:[unit]}
        ranks = circuit_kernel_magnitude_ranking(model, feature_targets = feature_targets)
        out = {}
        out['ranks'] = ranks
        out['layer'] = layer
        out['unit'] = unit
        out['method'] = 'magnitude'
        out['rank_field'] = None
        out['batch_size'] = None
        out['data_path'] =  None
        out['config'] = config_path
        torch.save(out,out_folder+str(layer)+':'+str(unit)+'_.pt')