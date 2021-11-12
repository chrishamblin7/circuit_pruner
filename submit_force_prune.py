from subprocess import call

config = 'configs/alexnet_sparse_config.py'
layers = ['features_6','features_8','features_10']
data_path = 'image_data/imagenet_2/'
Ts = [1,2,8]
units = range(20)
ratios = [.2,.005,.001]
device = 'cuda:1'

for layer in layers:
    for unit in units:
        feature_target = {layer:[unit]}
        for T in Ts:
            for ratio in ratios:
                call('python force_prune.py --T %s --ratio %s --unit %s --layer %s --config %s --data-path %s --device %s'%(str(T),str(ratio),str(unit),layer,config,data_path,device),shell=True)