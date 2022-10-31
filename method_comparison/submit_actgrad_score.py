
from subprocess import call

config = '../configs/resnet18_config.py'
# layers = ['mixed4a_5x5_pre_relu_conv','mixed4a_1x1_pre_relu_conv','mixed4a_3x3_pre_relu_conv','mixed4a_pool_reduce_pre_relu_conv',
#           'mixed4b_5x5_pre_relu_conv','mixed4b_1x1_pre_relu_conv','mixed4b_3x3_pre_relu_conv','mixed4b_pool_reduce_pre_relu_conv',
#           'mixed4d_5x5_pre_relu_conv','mixed4d_1x1_pre_relu_conv','mixed4d_3x3_pre_relu_conv','mixed4d_pool_reduce_pre_relu_conv',
#           'mixed3a_5x5_pre_relu_conv','mixed3a_1x1_pre_relu_conv','mixed3a_3x3_pre_relu_conv','mixed3a_pool_reduce_pre_relu_conv',
#           'mixed5a_5x5_pre_relu_conv','mixed5a_1x1_pre_relu_conv','mixed5a_3x3_pre_relu_conv','mixed5a_pool_reduce_pre_relu_conv',
#          ]
layers = ['layer2.0.conv1','layer2.0.conv2','layer2.1.conv1','layer2.1.conv2',
          'layer3.0.conv1','layer3.0.conv2','layer3.1.conv1','layer3.1.conv2',
          'layer4.0.conv1','layer4.0.conv2','layer4.1.conv1','layer4.1.conv2',]  

data_path = '../image_data/imagenet_2/'
device = 'cuda:2'

units = range(10)
out_root = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/circuit_ranks/'

batch_size = 3

for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)
        call_str = 'python actgrad_score.py --unit %s --layer %s --config %s --data-path %s --device %s --out-root %s --batch-size %s'%(str(unit),layer,config,data_path,device,out_root,str(batch_size))
        call(call_str,shell=True)