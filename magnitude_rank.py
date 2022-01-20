# generate kernel norm saliency scores for model





config = 'configs/alexnet_config.py'


units = list(range(20))
layers = ['features_3','features_6','features_8','features_10']



#get variables from config
if '/' in config:
    config_root_path = ('/').join(config.split('/')[:-1])
    update_sys_path(config_root_path)
config_module = config.split('/')[-1].replace('.py','')
params = __import__(config_module)

imageset = params.data_path.split('/')[-1]


model = params.model








#aggregate



#layer-normed





