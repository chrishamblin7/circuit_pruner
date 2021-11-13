from subprocess import call

config = 'configs/alexnet_sparse_config.py'
layers = ['features_3','features_6','features_8','features_10']
data_path = 'image_data/imagenet_2/'
units = range(20)
ratios = [.5,.2,.1,.05,.01,.005,.001]
device = 'cuda:1'

for layer in layers:
    print('PROCESSING LAYER: %s'%layer)
    for unit in units:
        print('PROCESSING UNIT: %s'%str(unit))
        feature_target = {layer:[unit]}

        call_str = 'python actgrad_prune.py --unit %s --layer %s --config %s --data-path %s --device %s'%(str(unit),layer,config,data_path,device)
        for r in ratios:
            call_str += ' --ratio %s '%str(r)
        call(call_str,shell=True)