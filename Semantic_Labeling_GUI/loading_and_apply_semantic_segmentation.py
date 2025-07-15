from model_architectures_for_semantic_segmentation import *
from dilution_check import *
from skimage import io, util, exposure, segmentation,measure, draw, morphology, restoration
from PIL import Image, ImageOps
from matplotlib.colors import ListedColormap
import os
import numpy as np
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

cuda = True if torch.cuda.is_available() else False


def load_model_semantic(load_dir,device='cpu'):
    state_dict=torch.load(load_dir,map_location=torch.device(device))
    
    class_names_semantic=state_dict['class_names_semantic']
    class_names_binary=state_dict['class_names_binary']
    complexity=state_dict['complexity']
    model_type=state_dict['model_type']
    n_low_res_layers=state_dict['n_low_res_layers']
    in_ch=state_dict['in_ch']
    out_ch_mask=state_dict['out_ch_mask']
    out_ch_semantic=state_dict['out_ch_semantic']
    separate_decoder=state_dict['separate_decoder']

    if model_type=='EnsembleSegNet':
        n_models=state_dict['n_models']
        geometric_model=state_dict['geometric_model']
        model=EnsembleSegNet(n_models=n_models,in_ch=in_ch,out_ch_mask=out_ch_mask,
                        out_ch_semantic=out_ch_semantic,complexity=complexity,
                        n_low_res_layers=n_low_res_layers,separate_decoder=separate_decoder,geometric_model=geometric_model,
                        class_names_semantic=class_names_semantic,class_names_binary=class_names_binary)
    elif model_type =='PseudoGeometricSegNet':
        n_models=state_dict['n_models']
        model=PseudoGeometricSegNet(n_models=n_models,in_ch=in_ch,out_ch_mask=out_ch_mask,
                        out_ch_semantic=out_ch_semantic,complexity=complexity,
                        n_low_res_layers=n_low_res_layers,separate_decoder=separate_decoder,
                        class_names_semantic=class_names_semantic,class_names_binary=class_names_binary)
    else:
        raise ValueError('Unknown model_architecture: ',model_architecture)

    
    model.load_state_dict(state_dict['model_state']) 
    
    if device == 'cpu':
        model.cpu()
    elif device == 'cuda':
        model.cuda()
    else:
        raise ValueError('device must be either cpu or cuda.')

    return model



def predict_phase_image(phase_img,model):
    model.eval()
    with torch.no_grad():
        out=model(phase_img)
        if isinstance(out, dict):
            logits=out
        else:
            logits=out[1]
        data_batch_hat={'object_mask':logits['binary_mask'][0].cpu(),
                        'semantic_mask':logits['semantic_mask'][0].cpu()}
    return data_batch_hat

def normalize_numpy_uint8_img(np_img,background_intensity=None):
    '''
    np_img: a uint8 valued numpy image of shape (width,height)
    estimates the background intensity and normalizes the image in a multiplicative way such that
    the image has background intensity of 0.5.
    returns: a numpy float image with background intensity normalized to 0.5
    '''
    if background_intensity is None:
        #estimate directly from np_img
        dilution_ok, background_ratio, background_intensity=is_dilution_ok(np_img.copy(), 
                                kernel_radius=1, threshold=0.05, prior_interval=None, squares_to_sample=25000, dilution_threshold=0.15)
    normalization_factor=(0.5/background_intensity)
    phase_img=np_img.astype('float')*(normalization_factor/255)#normalizing and casting to float approximately in [0,1]
    return phase_img,background_intensity

def preprocess_image(img_np,bg_int=None,cuda=False):
    if len(img_np.shape)==3:
        img_np = img_np[:,:,0]
    assert len(img_np.shape)==2, 'the image must be a one channel image with no dimension for the channel'
    img_np_rgb = np.repeat(np.expand_dims(img_np,axis=2),3,axis=2)
    img_np_normalized,bg_int = normalize_numpy_uint8_img(img_np,background_intensity=bg_int)
    img_torch = torch.from_numpy(img_np_normalized).unsqueeze(0).unsqueeze(0).float()
    if cuda:
        img_torch = img_torch.cuda()
    return img_torch, bg_int

import time

def get_raw_segmentation_output(img_np,model,bg_int=None,return_time_log = False,cuda=False):
    t0 = time.time()
    img_torch, bg_int = preprocess_image(img_np,bg_int,cuda=cuda)
    t1 = time.time()
    data_batch_hat = predict_phase_image(img_torch,model)
    t2 = time.time()
    segmentation_output = {'object_map_logits':data_batch_hat['object_mask'],
                          'semantic_map_logits':data_batch_hat['semantic_mask'],
                            'phase_img': img_np,
                            'background_intensity': bg_int,
                          }
    if return_time_log:
        time_log = {'t_preprocessing': t1 - t0, 't_model': t2-t1}
        return segmentation_output, time_log
    else:
        return segmentation_output


def count_parameters(model,trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())