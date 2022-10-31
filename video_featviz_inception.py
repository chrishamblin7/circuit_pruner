# from lucent_video.optvis import render, param, transform, objectives
# from PIL import Image
# from circuit_pruner.force import setup_net_for_circuit_prune, show_model_layer_names
# from circuit_pruner.force import mask_from_sparsity, expand_structured_mask, apply_mask
# from circuit_pruner.force import setup_net_for_circuit_prune, show_model_layer_names
# from circuit_pruner.utils import load_config
# from collections import OrderedDict
# import numpy as np
# import torch
# from math import ceil,exp,log
# import pickle
# import os
# from circuit_pruner.ranks import rankdict_2_ranklist
from circuit_pruner.simple_api.score import snip_score
from circuit_pruner.simple_api.mask import *
from lucent_video.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1
import torch
from circuit_pruner.utils import load_config
from circuit_pruner.data_loading import rank_image_data

from PIL import Image
from collections import OrderedDict
import numpy as np
from math import ceil,exp,log
import time



import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--layer", type = str, 
						help='layer_name')
	parser.add_argument('--unit', type=int,help='numeral for unit in layer of target feature')
	parser.add_argument("--config", type = str,default = 'configs/inception_config.py',help='relative_path to config file')
	parser.add_argument("--out-dir", type = str,default = './featureviz_videos/inception/',help='where to save output gifs')
	parser.add_argument('--device', type = str, default='cuda:0', help='default "cuda:0"')  
	parser.add_argument('--frames', type=int, default=600, help='how many frames in the gif')
	parser.add_argument('--opt-steps', type=int, default=150, help='how many times to run optimizer for each frame')
	parser.add_argument('--desaturation', type=float, default=4.0, help='desaturation param')
	parser.add_argument('--min-sparsity', type=float, default=.001, help='sparsity to start from')
	parser.add_argument('--max-sparsity', type=float, default=.4, help='sparsity to end at')


	args = parser.parse_args()
	return args


# device = 'cuda:0'
# out_dir = 'featureviz_videos/inception/'
# sparsity_range = (.001,.4)
# frames = 3
# opt_steps = 150
# desaturation = 4.0

# config_file = './configs/inception_config.py'

# targets = {'mixed4a':[501,499,492,476,475,460,391,323,288,287,269],
#             'mixed3b':[419,466,467],
#             'mixed4e':[779],
#             'mixed5a':[769]}



def gen_feature_viz_video(model,scores,layer,unit,
						  sparsities = None,sparsity_range = (.001,.6), frames=200,
						  scheduler='linear',connected=True,opt_steps=50,lr = 5e-2,size=224,
						  desaturation=10,file_name=None,save=True, negatives=True, include_full= True, reverse=False):
	
	setup_net_for_mask(model)
	
	if negatives:
		batch_size=4
		obj = objectives.channel(layer, unit, batch=0) + objectives.neuron(layer, unit, batch=1) - objectives.channel(layer, unit, batch=2) - objectives.neuron(layer, unit, batch=3)
	else:
		batch_size=2
		obj = objectives.channel(layer, unit, batch=0) + objectives.neuron(layer, unit, batch=1)

	image_list = []
	params = None
	opt = lambda params: torch.optim.Adam(params, lr)
	
	
	
	nonzero_scores = 0 
	for layer in scores:
		
		nonzero_scores += torch.sum((scores[layer] != 0).int())
	ks = []
	
	
	if sparsities is None:
		if scheduler == 'exp':

			max_k = int(nonzero_scores*sparsity_range[1])
			min_k = int(nonzero_scores*sparsity_range[0])
			for t in range(0,frames+1):
				k = ceil(exp(t/frames*log(min_k)+(1-t/frames)*log(max_k))) #exponential schedulr
				ks.insert(0, k) 
		else:
			sparsities = np.linspace(sparsity_range[0],sparsity_range[1],frames)
			for sparsity in sparsities:
				ks.append(int(nonzero_scores*sparsity))
	else:
		for sparsity in sparsities:
			ks.append(int(nonzero_scores*sparsity))
			
	#include_full
	if include_full and ks[-1] < nonzero_scores:
		ks.append(int(nonzero_scores))

	ks = list(OrderedDict.fromkeys(ks)) #remove duplicates
	for j in range(5):
		if j in ks:
			ks.remove(j)

	if reverse:
		ks.reverse()
	
	
	for k in ks:
		start = time.time()
		print(k)
		mask = mask_from_scores(scores,num_params_to_keep=k)
		#mask,cum_sal = mask_from_sparsity(ranks,k)
		#expanded_mask = expand_structured_mask(mask,model)
		apply_mask(model,mask)
		if connected:
			if params is not None:
				params = params.requires_grad_(False)
				##add noise
				#param_noise, _ = param.fft_image((1, 3, size, size),sd=.01)
				#param_noise = param_noise[0].requires_grad_(False)
				#params = param_noise+params
				
				#desaturate
				params = (params-torch.mean(params))/torch.std(params)*.05
				
				
				##remove high frequency 
				#params[:,:,:,10:,:] = 0
				
				params = params.requires_grad_(True)
				

			param_f = lambda: param.image(size,start_params=params,magic=desaturation,batch=batch_size)
		else:
			param_f = lambda: param.image(size,batch=batch_size)
		#import pdb; pdb.set_trace()
		output = render.render_vis(model, obj, param_f, opt, thresholds=(opt_steps,),progress=True,show_inline=True)
		image = output['images']
		params= output['params']
		
		image_list.append(image[0])
		
		#include full
		if include_full and (k ==ks[-1]) and not reverse:
			for _ in range(5):
				image_list.append(image[0])
				
		print(time.time()-start)

	#reshape
	for i in range(len(image_list)):
		im = image_list[i]
		if im.shape[0] > 1:
			image_list[i] = np.concatenate(tuple(im),axis=1)

	gif = []
	for image in image_list:
		im = Image.fromarray(np.uint8(image*255))
		gif.append(im)
	   
	if file_name is None:
		connected_name = ''
		if connected:
			connected_name = 'connected'
			file_name = '%s_%s_%s_paint.gif'%(layer,unit,connected_name)
		
	if save:
		print('saving gif to %s'%file_name)
		gif[0].save(file_name, save_all=True,optimize=False, append_images=gif[1:], loop=0)
	
	return image_list




def main():
	#args
	args = get_args()
	print(args)
	layer_name = args.layer
	unit = args.unit
	config_file = args.config
	out_dir = args.out_dir
	device = args.device
	frames = args.frames
	opt_steps = args.opt_steps
	desaturation = args.desaturation
	sparsity_range = (args.min_sparsity,args.max_sparsity)

	#model
	config = load_config(config_file)
	model = config.model.to(device)
	_ = model.to(device).eval()
	layers = OrderedDict([*model.named_modules()])


	#data loader
	kwargs = {'num_workers': 4, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}
	dataloader = torch.utils.data.DataLoader(rank_image_data(config.data_path,
											config.preprocess,
											label_file_path = config.label_file_path,class_folders=True),
											batch_size=64,
											shuffle=False,
											**kwargs)

	#for layer_name, target_units in targets.items():
	print(layer_name)
	#	for unit in target_units:
	print(unit)
	print('scoring')
	start = time.time()
	scores = snip_score(model,dataloader,layer_name,unit)
	print('score time: %s'%(str(time.time()-start)))
	start = time.time()
	_ = gen_feature_viz_video(model,scores,layer_name,unit,sparsity_range=sparsity_range,frames=frames,opt_steps=opt_steps,
	                file_name='%s/%s_%s.gif'%(out_dir,layer_name,str(unit)),scheduler='linear',connected=True, negatives=True,desaturation=desaturation, reverse=False)
	print('render time: %s'%(str(time.time()-start)))


if __name__ == '__main__':
	main()
