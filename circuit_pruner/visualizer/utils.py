import torch
import os
import argparse
import sys
import numpy as np
import pandas as pd
from circuit_pruner.utils import *
from circuit_pruner.ranks import *
from circuit_pruner.force import *


def gen_circuit_model_mapping_df(model,mask,ranks,version = 'edges'):
    
    ref_name_modules(model)
    layer_nodes = get_model_filterids(model)
    
    
    if version in ['kernels','edges']:
        kernel_ranks = structured_ranks_from_ranks(ranks,structure='kernels')
        normed_kernel_ranks = minmax_norm_ranks(kernel_ranks)
        kernel_mask = structured_mask_from_mask(mask)

        edge_columns = ['edge_num','layer_name','layer','out_channel','in_channel','rank','norm_rank']
        edge_biglist = []
        edge_num = -1
        for l, m in enumerate(kernel_mask):
            for out_channel in range(m.shape[0]):
                for in_channel in range(m.shape[1]):
                    edge_num += 1
                    if m[out_channel][in_channel] == 1:
                        #add edge info to df
                        layer_name = layer_nodes[l][0]
                        rank = float(kernel_ranks[l][out_channel][in_channel])
                        norm_rank = float(normed_kernel_ranks[l][out_channel][in_channel])
                        row = [edge_num,layer_name,l,out_channel,in_channel,rank,norm_rank]
                        edge_biglist.append(row)
                        
        return pd.DataFrame(edge_biglist,columns = edge_columns)

    else:
        filter_ranks = structured_ranks_from_ranks(ranks,structure='filters')
        normed_filter_ranks = minmax_norm_ranks(filter_ranks)  
        filter_mask = structured_mask_from_mask(mask,'filters')
        
        node_columns = ['node_num','layer_name','layer','node_num_by_layer','rank','norm_rank']
        node_biglist = []
        for l, m in enumerate(filter_mask):
            for filt in range(m.shape[0]):
                if m[filt] == 1:
                    #add node info to df
                    node_num = layer_nodes[l][1][filt]
                    layer_name = layer_nodes[l][0]
                    rank = float(filter_ranks[l][filt])
                    norm_rank = float(normed_filter_ranks[l][filt])
                    row = [node_num,layer_name,l,filt,rank,norm_rank]
                    node_biglist.append(row)
        
        return pd.DataFrame(node_biglist,columns = node_columns)
     



#### kernels/ edges  ####

def get_kernels_Conv2d_modules(module,kernels=[]): 
	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, torch.nn.modules.conv.Conv2d):
			kernels.append(submodule.weight.cpu().detach().numpy())
		elif len(list(submodule.children())) > 0:
			kernels = get_kernels_Conv2d_modules(submodule,kernels=kernels)   #module has modules inside it, so recurse on this module

	return kernels

#function for return a kernels inhibition/exhitation value, normalized between -1 and 1
def gen_kernel_posneg(kernels):
    kernel_colors = []
    for i, layer in enumerate(kernels):
        average = np.average(np.average(layer,axis=3),axis=2)
        absum = np.sum(np.sum(np.abs(layer),axis=3),axis=2)
        unnormed_layer_colors = average/absum
        #normalize layer between -1 and 1
        normed_layer_colors = 2/(np.max(unnormed_layer_colors)-np.min(unnormed_layer_colors))*(unnormed_layer_colors-np.max(unnormed_layer_colors))+1
        kernel_colors.append(normed_layer_colors)
    return kernel_colors

#function that takes kernel posneg values from -1 to 1 and returns rgba values
def posneg_to_rgb(kernel_posneg,color_anchors = [[10, 87, 168],[170,170,170],[194, 0, 19]]):
    
    #define a function for converting 'p' values between 0 and 1 to a 3 color vector
    color_anchors = np.array(color_anchors)
    def f(p,color_anchors=color_anchors):
        if p < .5:
            return np.rint(np.minimum(np.array([255,255,255]),color_anchors[1] * p * 2 +  color_anchors[0] * (0.5 - p) * 2))
        else:
            return np.rint(np.minimum(np.array([255,255,255]),color_anchors[2] * (p - 0.5) * 2 +  color_anchors[1] * (1 - p) * 2))
    #fnp = np.frompyfunc(f,1,1) 
    fnp = np.vectorize(f,signature='()->(n)') 

    kernel_colors = []
    for i, layer in enumerate(kernel_posneg):
        #nonlinear color interpolation
        ps = (layer+1)/2
        #ps = 1/(1+np.exp(-2*layer))
        kernel_colors.append(fnp(ps))
    return kernel_colors

'''
USING THE ABOVE FUNCTIONS:
kernels = get_kernels_Conv2d_modules(model)
kernel_posneg = gen_kernel_posneg(kernels)
kernel_colors = posneg_to_rgb(kernel_posneg)
'''



def circuit_edge_width_scaling(x):
	#return max(.4,(x*10)**1.7)
	return max(.5,np.exp(1.5*x))
	

def circuit_curve_2_id(curve_num,point_num,subgraph_dict,params):
	node_df = deepcopy(subgraph_dict['node_df'])
	node_df = node_df.sort_values(by=['node_num'])
	if curve_num == 0:
		imgnode_dict = {0:'r',1:'g',2:'b'}
		return imgnode_dict[point_num]
	elif curve_num <= len(node_df['layer'].unique()):
		layer = curve_num-1
		return str(node_df.loc[node_df['layer']==layer].iloc[point_num]['node_num'])
	else:
		edge_row_idx = curve_num - 1 - len(node_df['layer'].unique())
		row = subgraph_dict['edge_df'].iloc[edge_row_idx]
		if row['layer'] != 0:
			in_node = params['layer_nodes'][row['layer']-1][1][row['in_channel']]
		else:
			in_node = params['imgnode_names'][row['in_channel']]
		out_node = params['layer_nodes'][row['layer']][1][row['out_channel']]
		return str(in_node)+'-'+str(out_node)
	
