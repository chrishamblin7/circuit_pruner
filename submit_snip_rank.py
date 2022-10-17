from subprocess import call


config = 'configs/inception_config.py'
layers = ['conv2d2_pre_relu_conv','mixed5a_5x5_bottleneck_pre_relu_conv','mixed4e']
#layers = ['features_6']
data_path = 'image_data/imagenet_2/'
units = range(1)
device = 'cuda:0'


for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)
        feature_target = {layer:[unit]}
        call_str = 'python snip_rank.py --unit %s --layer %s --config %s --data-path %s --device %s'%(str(unit),layer,config,data_path,device)
        call(call_str,shell=True)


