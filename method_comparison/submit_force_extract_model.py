from subprocess import call

import time

start = time.time()

config = 'configs/alexnet_config.py'
layers = ['features_3','features_6','features_8','features_10']
#layers = ['features_6']
data_path = 'image_data/imagenet_2/'
units = range(20)
device = 'cuda:0'
sparsities = [.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.01,.005,.001]
#sparsities = [.95]

for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)
        for sparsity in sparsities:
            call_str = 'python force_extract_model.py --unit %s --layer %s --sparsity %s --config %s --data-path %s --device %s'%(str(unit),layer,str(sparsity),config,data_path,device)
            call(call_str,shell=True)


print(time.time()-start)
