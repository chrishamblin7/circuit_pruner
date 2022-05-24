### Config File for Alexnet Sparse. ###


import torch
import os
import sys
from circuit_pruner import root_path

### NAME ###

name = 'alexnet'


###MODEL###


from torchvision import models
import torch.nn as nn

model = models.alexnet(pretrained=True)


###DATA PATH###

if not os.path.exists(root_path+'/image_data/imagenet_2'):
	from circuit_pruner.download_from_gdrive import download_from_gdrive
	download_from_gdrive('alexnet_sparse',target = 'images')

data_path =  root_path+'/image_data/imagenet_2'   #Set this to the system path for the folder containing input images you would like to see network activation maps for.

label_file_path = root_path+'/image_data/imagenet_labels.txt'      #line seperated file with names of label classes as they appear in image names
						  #set to None if there are no target classes for your model
						  #make sure the order of labels matches the order in desired target vectors


