from lucent_video.optvis import render, param, transform, objectives
from PIL import Image
from circuit_pruner.force import setup_net_for_circuit_prune, show_model_layer_names
from circuit_pruner.force import mask_from_sparsity, expand_structured_mask, apply_mask
from circuit_pruner.force import setup_net_for_circuit_prune, show_model_layer_names
from circuit_pruner.utils import load_config
from collections import OrderedDict
import numpy as np
import torch
from math import ceil,exp,log
import pickle
import os
from circuit_pruner.ranks import rankdict_2_ranklist

device = 'cuda:0'
out_dir = 'featureviz_videos/alexnet_sparse/'
corr_thresh = .9
min_sparsity = .001
frames = 200
opt_steps = 100
rank_folder = '/mnt/data/chris/dropbox/Research-Hamblin/Projects/circuit_pruner/circuit_ranks/alexnet_sparse/imagenet_2/actxgrad/'
df_file = 'extracted_circuits/circuit_with_force_and_mag_df.pkl'
config_file = './configs/alexnet_sparse_config.py'


def gen_feature_viz_video(model,ranks,layer,unit,
                          sparsities = None,sparsity_range = (.001,.6), frames=200,
                          scheduler='linear',connected=True,opt_steps=50,lr = 5e-2,size=224,
                          desaturation=10,file_name=None,save=True, negatives=False, include_full= True):
    if negatives:
        batch_size=4
        obj = objectives.channel(layer, unit, batch=0) + objectives.neuron(layer, unit, batch=1) - objectives.channel(layer, unit, batch=2) - objectives.neuron(layer, unit, batch=3)
    else:
        batch_size=2
        obj = objectives.channel(layer, unit, batch=0) + objectives.neuron(layer, unit, batch=1)
    setup_net_for_circuit_prune(model)
    image_list = []
    params = None
    opt = lambda params: torch.optim.Adam(params, lr)
    
    
    
    nonzero_ranks = 0 
    for r in ranks:
        nonzero_ranks += torch.sum((r != 0).int())
    ks = []
    
    
    if sparsities is None:
        if scheduler == 'exp':

            max_k = int(nonzero_ranks*sparsity_range[1])
            min_k = int(nonzero_ranks*sparsity_range[0])
            for t in range(0,frames+1):
                k = ceil(exp(t/frames*log(min_k)+(1-t/frames)*log(max_k))) #exponential schedulr
                ks.insert(0, k) 
        else:
            sparsities = np.linspace(sparsity_range[0],sparsity_range[1],frames)
            for sparsity in sparsities:
                ks.append(int(nonzero_ranks*sparsity))
    else:
        for sparsity in sparsities:
            ks.append(int(nonzero_ranks*sparsity))
            
    #include_full
    if include_full and ks[-1] < nonzero_ranks:
        ks.append(int(nonzero_ranks))

    ks = list(OrderedDict.fromkeys(ks)) #remove duplicates
    for j in range(5):
        if j in ks:
            ks.remove(j)

    for k in ks:
        mask,cum_sal = mask_from_sparsity(ranks,k)
        expanded_mask = expand_structured_mask(mask,model)
        apply_mask(model,expanded_mask)
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
        image,params = render.render_vis(model, obj, param_f, opt, thresholds=(opt_steps,),return_params=True,progress=False,show_image=False)
        image_list.append(image[0])
        
        #include full
        if include_full and (k ==ks[-1]):
            for _ in range(5):
                image_list.append(image[0])

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
    df = pickle.load(open(df_file,'rb'))
    config = load_config(config_file)
    model = config.model.to(device)

    rank_files = os.listdir(rank_folder)

    for rank_file in rank_files:
        ranks = torch.load(rank_folder+rank_file)
        ranks = rankdict_2_ranklist(ranks)
        unit = int(rank_file.split(':')[-1].split('_')[0])
        layer = rank_file.split(':')[0].replace('alexnet_sparse_','')


        if os.path.exists('%s/%s_%s_linear.gif'%(out_dir,layer,str(unit))):
            print('skipping %s_%s_linear.gif'%(layer,str(unit)))
            continue
        selection = df.loc[(df['model']=='alexnet_sparse') & (df['layer']==layer) & (df['method']=='actxgrad') & (df['unit']==unit)]
        sparsities = list(selection['sparsity'])
        scores = list(selection['pruned_pearson'])

        for i in range(len(scores)):
            if scores[i] > corr_thresh:
                upper = scores[i]
                under = scores[i-1]
                upper_i = sparsities[i]
                under_i = sparsities[i-1]
                upper_sparsity = (corr_thresh - under)*(upper_i-under_i)/(upper-under)+under_i
                break
            
        
        print(layer+':'+str(unit)+'    '+str(upper_sparsity))

        _ = gen_feature_viz_video(model,ranks,layer,unit,sparsity_range=(min_sparsity,upper_sparsity),frames=frames,opt_steps=opt_steps,
                            file_name='%s/%s_%s_linear.gif'%(out_dir,layer,str(unit)),scheduler='linear',connected=True, negatives=False)



if __name__ == '__main__':
    main()
