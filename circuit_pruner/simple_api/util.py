import torch
from torch import nn
from collections import OrderedDict
from math import ceil


def params_2_target_from_scores(scores,unit,target_layer_name,model):
    #total params
    all_layers = OrderedDict([*model.named_modules()])

    total_params = 0
    for layer_name, layer_scores in scores.items():
        layer = all_layers[layer_name]
        if layer_name == target_layer_name or layer_name == next(reversed(scores)): #not all weights matter, only those leading to target
            if isinstance(unit,int):
                dims = 1
            else:
                dims = torch.sum(torch.tensor(unit) != 0)
            #EDIT this might not be general
            total_params += int(torch.numel(layer.weight)/layer.weight.shape[0]*dims) 
        else:
            total_params += torch.numel(layer_scores)

    return total_params

def params_2_target_in_layer(unit,layer):
    '''
    how many parameters in the target layer actually attach to the target?
    Useful for getting relevant sparsity measure
    EDIT: Is this really general? Does it work with a branching layer for example?
    '''
    if isinstance(unit,int):
        dims = 1
    else:
        dims = torch.sum(torch.tensor(unit) != 0)
    return int(torch.numel(layer.weight)/layer.weight.shape[0]*dims)   

def get_layers(model, parent_name='', layer_info=[]):
    for module_name, module in model.named_children():
        layer_name = parent_name + '.' + module_name
        if len(list(module.named_children())):
            layer_info = get_layers(module, layer_name, layer_info=layer_info)
        else:
            layer_info.append(layer_name.strip('.'))
    
    return layer_info

def get_layer_names(model):
    return get_layers(model, parent_name='', layer_info=[])

def get_layer_type(model, layer_name):
    for name,m in list(model.named_modules()):
        if name == layer_name: return m.__class__.__name__
            
def convert_relu_layers(model):
  #name should be the module name according to the 'OrderedDict([*model.named_modules()])' method ("." nesting)
  #useful for doing things like changing a relu to 'inplace'

    # recursive function to get layers
    def get_layers(module):
        if hasattr(module, "_modules"):
            for name, layer in module._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if isinstance(layer, nn.ReLU):
                  layer = nn.ReLU(inplace=False)
                  setattr(module, name, nn.ReLU(inplace=False))
                
                setattr(module, name, layer)
                get_layers(layer)

    get_layers(model)



def inplace_model_edit(model, target_name, new_module):
  #name should be the module name according to the 'OrderedDict([*model.named_modules()])' method ("." nesting)
  #useful for doing things like changing a relu to 'inplace'

    # recursive function to get layers
    def get_layers(module, prefix=[]):
        if hasattr(module, "_modules"):
            for name, layer in module._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                full_name = ".".join(prefix+[name])
                if full_name == target_name:
                  layer = new_module
                  setattr(module, name, new_module)
                
                setattr(module, name, layer)
                get_layers(layer, prefix=prefix+[name])

    get_layers(model)