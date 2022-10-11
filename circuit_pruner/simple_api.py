import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, Iterable, Callable
from collections import OrderedDict
from circuit_pruner.custom_exceptions import TargetReached
import types
from copy import deepcopy




####  RANKING ####

def snip_rank(model,dataloader,layer_name,unit,layer_types_2_rank = [nn.Conv2d,nn.Linear]):
    
    device = next(model.parameters()).device  
    layers = OrderedDict([*model.named_modules()])

    target_saver = feature_target_saver(model,layer_name,unit)
    ranks = OrderedDict()  
    for i, data in enumerate(dataloader, 0):

        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)

        model.zero_grad() #very import!
        try:
            output = model(inputs)
        except TargetReached:
            pass
            
        #feature collapse, this has saved all activation values for our feature we need to collapse into a single number
        #Here well just use the sum of the absolute values
        loss = torch.sum(torch.abs(target_saver.target))
        loss.backward()

        
        #get weight-wise scores
        for layer_name,layer in layers.items():
            if type(layer) not in layer_types_2_rank:
                continue
                
            if layer.weight.grad is None:
                continue
            
            if layer not in ranks.keys():
                ranks[layer_name] = torch.abs(layer.weight*layer.weight.grad).detach().cpu()
            else:
                ranks[layer_name] += torch.abs(layer.weight*layer.weight.grad).detach().cpu()
                
    # # eliminate layers with stored but all zero ranks
    #print('layers eliminated with zero rank')
    remove_keys = []
    for layer in ranks:
        if torch.sum(ranks[layer]) == 0.:
            print(layer)
            remove_keys.append(layer)
    for k in remove_keys:
        del ranks[k]
          
    target_saver.hook.remove() # this is important or we will accumulate hooks in our model

    return ranks




#### Target   #####


class feature_target_saver(nn.Module):
    '''
    takes a model, and adds a target
    layer: str name of layer, use "dict([*self.model.named_modules()])"
    unit: int or list/array the length of the out features dim of the layer (specifying coefficients)
    '''


    def __init__(self, model, layer, unit, kill_forward = True):
        super().__init__()
        self.model = model
        self.layer = layer
        self.unit = unit
        self.target = None
        layer = OrderedDict([*self.model.named_modules()])[layer]
        self.hook = layer.register_forward_hook(self.get_target()) #execute on forward pass

        self.kill_forward = kill_forward

    def get_target(self) -> Callable:
        def fn(module, input, output):  #register_hook expects to recieve a function with arguments like this
            #output is what is return by the layer with dim (batch_dim x out_dim), sum across the batch dim
            if isinstance(self.unit,int):
                target_features = output[:,self.unit]
            else:
                assert len(self.unit) == output.shape[1]
                self.unit = torch.tensor(self.unit).to(output.device).type(output.dtype)
                target_features = torch.tensordot(output, self.unit, dims=([1],[0]))

            self.target = target_features

            #import pdb; pdb.set_trace()

            if self.kill_forward:
                #print('feature target in %s reached.'%self.layer)
                raise TargetReached
            #aggregrate across batches? seems confusing . . .
            #if self.target is None:
            #    self.target = target_features
            #else:
            #    self.target = torch.cat((self.target,target_features),dim=0)

        return fn
    




#### MASKS #####


def mask_from_ranks(ranks,sparsity=None,num_params_to_keep=None):
    assert not ((sparsity is None) and (num_params_to_keep is None))

    keep_masks = {}
    
    #flatten
    ranks_flat = torch.cat([torch.flatten(x) for x in ranks.values()])
    norm_factor = torch.sum(abs(ranks_flat))
    ranks_flat.div_(norm_factor)

    #num kept params
    if not num_params_to_keep is None:
        k = num_params_to_keep
    else:
        k = int(len(ranks_flat) * sparsity)

    #get threshold score
    threshold, _ = torch.topk(ranks_flat, k, sorted=True)
    acceptable_rank = threshold[-1]


    
    if acceptable_rank == 0:
        print('gradients from this feature are sparse,\
                the minimum acceptable rank at this sparsity has a score of zero! \
                we will return a mask thats smaller than you asked, by masking all \
                parameters with a score of zero.')

    for layer_name in ranks:
        layer_ranks = ranks[layer_name]
        keep_masks[layer_name] = (layer_ranks / norm_factor > acceptable_rank).float()
    
    return keep_masks


def masked_conv2d_forward(self, x):

    #pass input through conv and weight mask

    x = F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups) 

    return x

def masked_linear_forward(self, x):

    x = F.linear(x, self.weight * self.weight_mask, self.bias)

    return x

def setup_net_for_mask(model):

    #same naming trick as before 
    layers = OrderedDict([*model.named_modules()])

    for layer in layers.values():
        if isinstance(layer, nn.Conv2d):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.forward = types.MethodType(masked_conv2d_forward, layer)
        elif isinstance(layer, nn.Linear):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.forward = types.MethodType(masked_linear_forward, layer)
            
def apply_mask(model,mask, zero_absent=True):
    layers = OrderedDict([*model.named_modules()])
    for layer_name in mask:
        layers[layer_name].weight_mask = nn.Parameter(mask[layer_name].to(layers[layer_name].weight.device))
    if zero_absent:
        #mask all layers not specified in the mask
        for layer_name in layers:
            if layer_name not in mask.keys():
                try:
                    layers[layer_name].weight_mask = nn.Parameter(torch.zeros_like(layers[layer_name].weight))
                except:
                    pass


def mask_intersect_over_union(mask1,mask2):
    iou = {}
    for layer_name in mask1:
        intersect_mask = mask1[layer_name]*mask2[layer_name]
        union_mask = torch.ceil((mask1[layer_name]+mask2[layer_name])/2)
        iou[layer_name] = (torch.sum(intersect_mask)/torch.sum(union_mask))
    return iou



#### Rank manipulations ####

def structure_ranks(ranks, model, structure='kernels'):
    
    assert structure in ['kernels','filters']
    layers = OrderedDict([*model.named_modules()])
    
    if structure == 'kernels':
        collapse_dims = (2,3)
    else:
        collapse_dims = (1,2,3)
        
    structured_ranks = {}
    for layer_name in ranks:
        if isinstance(layers[layer_name],nn.Conv2d):
            structured_ranks[layer_name] = torch.mean(ranks[layer_name],dim=collapse_dims)
            
    return structured_ranks      
        

def minmax_norm_ranks(ranks, min=0, max=1):
    out_ranks = {}
    for layer_name, scores in ranks.items():
        old_min = torch.min(scores)
        old_max = torch.max(scores)
        out_ranks[layer_name] = (scores - old_min)/(old_max - old_min)*(max - min) + min

    return out_ranks




#### Diagnositic #####
def get_cumulative_saliency_per_sparsity(ranks,step_size=None,window=(0,1)):

    ranks_flat = torch.cat([torch.flatten(x) for x in ranks.values()])
    #norm_factor = torch.sum(abs(ranks_flat))
    #ranks_flat.div_(norm_factor)
    ranks_sorted = torch.sort(ranks_flat,descending=True).values

    total_weight = torch.sum(abs(ranks_sorted))

    if step_size is None:
        step_size = int(len(ranks_flat)/500)
        print('step_size not specified, choosing a size of %s (weights), \
such that there are 500 sparsity levels measured'%str(step_size))

    window = [int(window[0]*len(ranks_sorted)),int(window[1]*len(ranks_sorted))]

    sparsities = []
    cum_weights = []

    for i in range(window[0],window[1],step_size):
        sparsities.append((i/len(ranks_sorted)))
        
        #cum_weight = torch.sum(ranks_sorted[i-step_size:i+1])
        #cum_weights.append(float(cum_weight))
        cum_weight = torch.sum(ranks_sorted[:i+1])
        cum_weights.append(float(cum_weight/total_weight))

    return sparsities,cum_weights


def get_num_param_at_cumulative_rank(ranks,cum_rank=.95):

    ranks_flat = torch.cat([torch.flatten(x) for x in ranks.values()])
    #norm_factor = torch.sum(abs(ranks_flat))
    #ranks_flat.div_(norm_factor)
    ranks_sorted = torch.sort(ranks_flat,descending=True).values

    total_weight = torch.sum(abs(ranks_sorted))

    if step_size is None:
        step_size = int(len(ranks_flat)/500)
        print('step_size not specified, choosing a size of %s (weights), \
such that there are 500 sparsity levels measured'%str(step_size))

    window = [int(window[0]*len(ranks_sorted)),int(window[1]*len(ranks_sorted))]

    sparsities = []
    cum_weights = []

    for i in range(window[0],window[1],step_size):
        sparsities.append((i/len(ranks_sorted)))
        
        #cum_weight = torch.sum(ranks_sorted[i-step_size:i+1])
        #cum_weights.append(float(cum_weight))
        cum_weight = torch.sum(ranks_sorted[:i+1])
        cum_weights.append(float(cum_weight/total_weight))

    return sparsities,cum_weights


 





#### EXTRA  #### (Probably to be unused and should be moved)


class actgrad_extractor(nn.Module):
    '''
    scores features (nodes), not weights (edges)
    '''
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.activations = {layer: None for layer in layers}
        self.gradients = {layer: None for layer in layers}
        self.hooks = {'forward':{},
                 'backward':{}}   #saving hooks to variables lets us remove them later if we want
        
        for layer_id in layers:
            layer = OrderedDict([*self.model.named_modules()])[layer_id]
            self.hooks['forward'][layer_id] = layer.register_forward_hook(self.save_activations(layer_id)) #execute on forward pass
            self.hooks['backward'][layer_id] = layer.register_backward_hook(self.save_gradients(layer_id))    #execute on backwards pass

    def save_activations(self, layer_id: str) -> Callable:
        def fn(module, input, output):  #register_hook expects to recieve a function with arguments like this
            #output is what is return by the layer with dim (batch_dim x out_dim), sum across the batch dim
            batch_summed_output = torch.sum(torch.abs(output),dim=0).detach().cpu()
            if self.activations[layer_id] is None:
                self.activations[layer_id] = batch_summed_output
            else:
                self.activations[layer_id] +=  batch_summed_output
        return fn
    
    def save_gradients(self, layer_id: str) -> Callable:
        def fn(module, grad_input, grad_output):
            batch_summed_output = torch.sum(torch.abs(grad_output[0]),dim=0).detach().cpu() #grad_output is a tuple with 'device' as second item
            if self.gradients[layer_id] is None:
                self.gradients[layer_id] = batch_summed_output
            else:
                self.gradients[layer_id] +=  batch_summed_output 
        return fn
    
    def remove_all_hooks(self):
        for hook in self.hooks['forward'].values():
            hook.remove()
        for hook in self.hooks['backward'].values():
            hook.remove()