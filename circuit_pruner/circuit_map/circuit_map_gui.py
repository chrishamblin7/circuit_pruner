#loading base packages
import os
import torch
from PIL import Image
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torchvision
import pandas as pd
import pickle
from copy import deepcopy

#library specific packages
from circuit_pruner.simple_api.target import sum_abs_loss, positional_loss, feature_target_saver
from circuit_pruner.simple_api.mask import setup_net_for_mask, mask_from_scores, apply_mask
from circuit_pruner.simple_api.util import params_2_target_from_scores
from circuit_pruner.simple_api.score import actgrad_filter_score, actgrad_kernel_score, get_num_params_from_cum_score, snip_score
from circuit_pruner.data_loading import single_image_data
from lucent_video.optvis import render, param, transform, objectives
from lucent_video.optvis.render_video import render_accentuation
from circuit_pruner.simple_api.dissected_Conv2d import dissect_model

#plotting
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go
import io
import base64
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix




def image_path_to_base64(im_path,pos=None,size = (input_size[1],input_size[2])):
    img = Image.open(im_path)
    if pos is not None:
        img = position_crop_image(img,pos,layer_name)
    else:
        img = img.resize(size)
    buffer = io.BytesIO()
    img.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def pil_image_to_base64(img,pos=None,size = (input_size[1],input_size[2])):
    if pos is not None:
        img = position_crop_image(img,pos,layer_name)
    else:
        img = img.resize(size)
    buffer = io.BytesIO()
    img.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url




def umap_fig_from_df(df,data_folder=None,normed=False,align_df=None,num_display_images=50,act_column = None,color_std=None,show_colorscale=True):
    '''
    df: a umap df
    data_folder: path to images
    align_df: df to rotationally align to (usually done before hand)
    '''
    fig = go.Figure()
    #positions
    xy_addition = ''
    if normed:
        xy_addition = '_normed'
    x = list(df['x'+xy_addition])
    y = list(df['y'+xy_addition])
    #norms
    norms = list(df['norm'])
    #activations
    act_column = act_column or 'activation'
    acts = list(df[act_column]) 
    
    #color
    color_std = color_std or torch.std(torch.tensor(acts)) 
    color_limit = float(color_std*3)

    fig.add_trace(go.Scatter(
      x=x,
      y=y,
      marker=dict(
                  line=dict(width=.5,
                            color='Grey'),
                  cmid=0,
                  cmin=-color_limit,
                  cmax=color_limit,
                  size=torch.tensor(norms)/torch.mean(torch.tensor(norms))*3.5,
                  color=acts,
                  colorbar=dict(
                                  title="Activation"
                              ),
                  colorscale="RdBu_r"
                  ),
      mode="markers",
      name='points'
      ))
    
    
    #alignment trace
    if align_df is not None:
        #add line figs
        x_align = list(align_df['x'+xy_addition])
        y_align = list(align_df['y'+xy_addition])
        x_joint = []
        y_joint = []
        for i in range(len(x)):
            x_joint.append(x_align[i])
            x_joint.append(x[i])
            x_joint.append(None)
            y_joint.append(y_align[i])
            y_joint.append(y[i])
            y_joint.append(None)

        fig.add_trace(go.Scatter(
                                x=x_joint, 
                                y=y_joint,
                                line=dict(color='grey', width=.5),
                                mode='lines',
                                name='alignment',
                                visible='legendonly'))

        
    if show_colorscale:
        fig.update_traces(marker_showscale=False)
    
    layout = go.Layout(   margin = dict(l=5,r=5,b=5,t=5),
                          paper_bgcolor='rgba(255,255,255,1)',
                          plot_bgcolor='rgba(255,255,255,1)',
                          xaxis=dict(showline=False,showgrid=False,showticklabels=False,range=[torch.min(torch.tensor(x))-1, torch.max(torch.tensor(x))+1]),
                          yaxis=dict(showline=False,showgrid=False,showticklabels=False,scaleanchor="x", scaleratio=1,range=[torch.min(torch.tensor(y))-1, torch.max(torch.tensor(y))+1]))

    fig.layout = layout
    

    #images
    if (data_folder is not None) and num_display_images>0:
        
        #select images far apart
        pts2D = np.swapaxes(np.array([list(df['x']),list(df['y'])]),0,1)
        kmeans = KMeans(n_clusters=num_display_images, random_state=0).fit(pts2D)
        labels = kmeans.predict(pts2D)
        cntr = kmeans.cluster_centers_
        approx = []
        for i, c in enumerate(cntr):
            lab = np.where(labels == i)[0]
            pts = pts2D[lab]
            d = distance_matrix(c[None, ...], pts)
            idx1 = np.argmin(d, axis=1) + 1
            idx2 = np.searchsorted(np.cumsum(labels == i), idx1)[0]
            approx.append(idx2)
        #add layout images
        use_position=None
        if 'position' in df.columns:
            use_position = True
        for i in approx:
            position=None
            if use_position:
                position = df.iloc[i]['position']  
            img = image_path_to_base64(data_folder+'/'+df.iloc[i]['image'],pos=position,size = (input_size[1],input_size[2]))
            #img = Image.open(data_folder+'/'+df.iloc[i]['image']) 
            fig.add_layout_image(
                                dict(
                                    source=img,
                                    #source="http://chrishamblin.xyz/images/viscnn_images/%s.jpg"%nodeid,
                                    xref="x",
                                    yref="y",
                                    x=df.iloc[i]['x'+xy_addition],
                                    y=df.iloc[i]['y'+xy_addition],
                                    sizex=.5,
                                    sizey=.5,
                                    xanchor="center",
                                    yanchor="middle",
                                    layer='above',
                                    opacity=.5
                                ))
    

    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    return fig



def full_app_from_df(df,data_folder,normed=True, align_df = None,max_images=200):
    use_position = False
    if 'position' in df.columns:
        use_position = True
    
    top_row = df[df['activation'] == df['activation'].max()].iloc[0]
    top_row_pos = None
    if use_position:
        top_row_pos = top_row['position']
    start_image = image_path_to_base64(data_folder+top_row['image'],pos=top_row_pos)

    umap_fig = umap_fig_from_df(df,normed=normed, align_df=align_df)
    
    xy_addition = ''
    if normed:
        xy_addition = '_normed'
    #image order
    #select images far apart
    pts2D = np.swapaxes(np.array([list(df['x']),list(df['y'])]),0,1)
    kmeans = KMeans(n_clusters=max_images, random_state=0).fit(pts2D)
    labels = kmeans.predict(pts2D)
    cntr = kmeans.cluster_centers_
    image_order = []
    for i, c in enumerate(cntr):
        lab = np.where(labels == i)[0]
        pts = pts2D[lab]
        d = distance_matrix(c[None, ...], pts)
        idx1 = np.argmin(d, axis=1) + 1
        idx2 = np.searchsorted(np.cumsum(labels == i), idx1)[0]
        image_order.append(idx2)
    #all layout images
    all_layout_images = []        
    for i in image_order:
        position=None
        if use_position:
            position = df.iloc[i]['position']  
        img = image_path_to_base64(data_folder+'/'+df.iloc[i]['image'],pos=position,size = (input_size[1],input_size[2]))
        #img = Image.open(data_folder+'/'+df.iloc[i]['image']) 
        all_layout_images.append(
                                    dict(
                                        source=img,
                                        #source="http://chrishamblin.xyz/images/viscnn_images/%s.jpg"%nodeid,
                                        xref="x",
                                        yref="y",
                                        x=df.iloc[i]['x'+xy_addition],
                                        y=df.iloc[i]['y'+xy_addition],
                                        sizex=.5,
                                        sizey=.5,
                                        xanchor="center",
                                        yanchor="middle",
                                        layer='above',
                                        opacity=.8
                                    ))


    #external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    #app = JupyterDash(__name__,external_stylesheets = external_stylesheets)
    app = JupyterDash(__name__)

    colors = {
            'background': 'rgba(220,220,220,1)',
            'text': '#111111'
            }

    app.layout = html.Div([

      html.Div([
        #dcc.Graph(id="umap", figure=umap_fig, clear_on_unhover=True,style={'width': '100vh', 'height': '100vh'}),
        dcc.Graph(id="umap", figure=umap_fig, clear_on_unhover=True),
        html.Label('# plot images'),
        dcc.Slider(0, max_images, 1,
                  marks={i: str(i) for i in range(0,max_images,10)},
                  value=0,
                  id='num_images_slider'
                 ),
        dcc.Tooltip(id="graph-tooltip")
               ],style={'width': '49%','display': 'inline-block'}),
        
      html.Br(),
        
      html.Div([
        html.Img(src=start_image, id='click-image'),
        html.Img(src=start_image, id='accent-image-max'),
        html.Img(src=start_image, id='noise-image-max'),
        html.Img(src=start_image, id='accent-image-min'),
        html.Img(src=start_image, id='noise-image-min'),
        html.Br()
                ],style={'width': '90%','display': 'inline-block'}),
        
      html.Div([
        html.Label('accentuation steps'),
        dcc.Slider(0, 40, 1,
                  value=20,
                  marks={i: str(i) for i in range(0,40,5)},
                  id='accent_threshold_slider'
                 ),
        html.Label('saturation'),
        dcc.Slider(0, 1, .01,
                  value=.5,
                  marks={i*.1: '{}'.format(round(i*.1,1)) for i in range(10)},
                  id='saturation_slider'
                 ),
        html.Label('cumulative weight in model'),
        dcc.Slider(.5, 1, .005,
                  value=.98,
                  marks={.5+ i*.05: '{}'.format(round(.5+ i*.05,2)) for i in range(20)},
                  updatemode='drag',
                  id='accent_sparsity_slider',
                 ),
        html.Br(),
        html.Label('pruning type'),
        dcc.RadioItems(['filters', 'kernels', 'weights'],'filters',id='pruning_type'),
        html.Div(id='sparsity'),
      ],style={'width': '49%','display': 'inline-block'}),
        
      #background
      dcc.Store(id='memory')
      ],style={'backgroundColor':colors['background'],'color':colors['text']})


    @app.callback(
      Output("graph-tooltip", "show"),
      Output("graph-tooltip", "bbox"),
      Output("graph-tooltip", "children"),
      Input("umap", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        df_row = df.iloc[num]
        
        position=None
        if use_position:
            position = df_row['position']
        
        img_path = data_folder+'/'+df_row['image']
        img_src = image_path_to_base64(img_path,pos=position)
        act = round(df_row['activation'],3)
        norm = round(df_row['norm'],3)

        children = [
          html.Div(children=[
                              html.Img(src=img_src, style={"width": "100%"}),
                              html.P(f"activation: {act}"),
                              html.P(f"norm: {norm}"),
                              html.P(f"Image: {df_row['image']}"),
                              html.P(f"Position: {position}")
                              ],style={'width': '200px', 'white-space': 'normal'})
                    ]

        return True, bbox, children


    @app.callback(
    Output("memory", "data"),
    Input("umap", "clickData"),
    Input("accent_sparsity_slider",'value'),
    Input("saturation_slider",'value'),
    Input("pruning_type","value")
    )
    def store_images(clickData,cum_score,sat,pruning_type):

        # demo only shows the first point, but other points may also be available
        pt = clickData["points"][0]
        num = pt["pointNumber"]

        df_row = df.iloc[num]
        img_path = data_folder+'/'+df_row['image']
        
        position=None
        loss = sum_abs_loss
        if use_position:
            position = df_row['position']
            loss = positional_loss(position)


        #get scores
        setup_net_for_mask(model)
        
        sparsity = 1.
        if cum_score < 1:
            dataloader = DataLoader(single_image_data(img_path,
                                                      preprocess),
                                    batch_size=1,
                                    shuffle=False
                                    )
            if pruning_type == 'weights':
                scores = snip_score(model,dataloader,layer_name.replace('_','.'),unit,loss_f=loss)
            elif pruning_type == 'kernels':
                scores = actgrad_kernel_score(dis_model,dataloader,layer_name.replace('_','.'),unit,loss_f=loss,dissect_model=False)
            else:
                scores = actgrad_filter_score(model,dataloader,layer_name.replace('_','.'),unit,loss_f=loss)  
            keep_params = get_num_params_from_cum_score(scores,cum_score)
            total_params = params_2_target_from_scores(scores,unit,layer_name,model)
            sparsity = keep_params/total_params
            mask = mask_from_scores(scores, num_params_to_keep = keep_params)
            apply_mask(model,mask,zero_absent=False) #dont zero absent as scores dont have target layer



        orig_pil_img = Image.open(img_path)
        orig_img_src = pil_image_to_base64(orig_pil_img,pos=position)
        accent_output = render_accentuation(img_path,layer_name,unit,model,saturation=sat*16.,device=device,size=224,show_image=False)

        data = {'images':{'orig':orig_img_src,
                        'max':{},
                        'min':{},
                        'min_noise':{},
                        'max_noise':{}},
                 'sparsity':round(sparsity,3)}

        for frame, frame_image in enumerate(accent_output['images']):
            all_accent_tensor_img = frame_image
            accent_tensor_img_max = all_accent_tensor_img[0]
            accent_tensor_img_min = all_accent_tensor_img[1]
            noise_tensor_img_max = all_accent_tensor_img[2]
            noise_tensor_img_min = all_accent_tensor_img[3]
            accent_img_max = Image.fromarray(np.uint8(accent_tensor_img_max*255))
            accent_img_min = Image.fromarray(np.uint8(accent_tensor_img_min*255))
            noise_img_max = Image.fromarray(np.uint8(noise_tensor_img_max*255))
            noise_img_min = Image.fromarray(np.uint8(noise_tensor_img_min*255))
            data['images']['max']['frame %s'%(str(frame))] = pil_image_to_base64(accent_img_max,pos=position)
            data['images']['min']['frame %s'%(str(frame))] = pil_image_to_base64(accent_img_min,pos=position)
            data['images']['max_noise']['frame %s'%(str(frame))] = pil_image_to_base64(noise_img_max,pos=position)
            data['images']['min_noise']['frame %s'%(str(frame))] = pil_image_to_base64(noise_img_min,pos=position)

        return data



    @app.callback(
      Output("click-image", "src"),
      Output('accent-image-max','src'),
      Output('accent-image-min','src'),
      Output('noise-image-max','src'),
      Output('noise-image-min','src'),
      Input("memory", "data"),
      Input("accent_threshold_slider", "value"),
    )
    def display_click_images(memory,frame):

        frame = int(frame)
        return memory['images']['orig'], memory['images']['max']['frame '+str(frame)], memory['images']['min']['frame '+str(frame)], memory['images']['max_noise']['frame '+str(frame)],memory['images']['min_noise']['frame '+str(frame)]

    
    @app.callback(
        Output('sparsity', 'children'),
        Input('memory', 'data')
    )
    def update_sparsity(memory):
        return 'Sparsity: %s'%str(memory["sparsity"])
     
        
        
    @app.callback(
                  Output("umap", "figure"),
                  Input("num_images_slider", "value"),
                  State("umap", "figure"),
                )
    def display_fig_images(num_images,fig):
        try:
            current_images = len(fig['layout']['images'])
        except:
            current_images = 0
            fig['layout']['images'] = tuple()
        if num_images <=current_images:
            fig['layout']['images'] = fig['layout']['images'][:num_images]
            return fig
        new_images = list(fig['layout']['images'])
        for k in range(current_images,num_images):
            new_images.append(all_layout_images[k])
        fig['layout']['images'] = tuple(new_images)
        return fig
    
    return app

