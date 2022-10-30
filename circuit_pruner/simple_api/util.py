import torch

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
            
def convert_relu_layers(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU) and child.inplace==True:
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu_layers(child)