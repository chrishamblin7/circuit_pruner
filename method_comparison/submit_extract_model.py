from subprocess import call

import time

start = time.time()

config = '../configs/alexnet_sparse_config.py'
layers = ['features_3','features_6','features_8','features_10']
#data_path = '../image_data/imagenet_2/'
data_path = '../image_data/imagenet_2_test/'
rank_data_path = '../image_data/imagenet_2/'
units = range(20)
device = 'cuda:1'
method = 'snip'
sparsities = [.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.01,.005,.001]


for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)
        for sparsity in sparsities:
            call_str = 'python extract_model.py --method %s --unit %s --layer %s --sparsity %s --config %s --data-path %s --rank-data-path %s --device %s'%(method,str(unit),layer,str(sparsity),config,data_path,rank_data_path,device)
            call(call_str,shell=True)


print(time.time()-start)
