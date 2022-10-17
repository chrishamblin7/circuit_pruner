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
    


### Losses ### 
'''
The feature target saver saves something multidimensional, but we have to 
backprop 'losses', basically scalars. Here are different reasonable functions
for collapsing a target feature into a scalar, that can be back propogated.
'''

def sum_loss(target):
    return torch.sum(target)

def sum_abs_loss(target):
    return torch.sum(torch.abs(target))




