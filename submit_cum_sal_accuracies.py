from subprocess import call

config = 'configs/alexnet_sparse_config.py'
layers = ['features_3','features_6','features_8','features_10']
data_path = 'image_data/imagenet_2/'
units = range(20)
device = 'cuda:0'
method = 'snip'
by_sparsity = True

for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)

        feature_target = {layer:[unit]}

        call_str = 'python cum_sal_accuracies.py --method %s --unit %s --layer %s --config %s --data-path %s --device %s'%(method,str(unit),layer,config,data_path,device)
        if by_sparsity:
            call_str += ' --by-sparsity'

        call(call_str,shell=True)