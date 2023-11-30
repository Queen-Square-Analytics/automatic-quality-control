import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision import models
from PIL import Image


def prepare_input(img, shape):
    img = Image.fromarray(img)
    img = TVF.resize(img, shape)
    img = TVF.to_tensor(img)
    img = TVF.normalize(img, [.5], [.5])
    return img.unsqueeze(0)


def critic_features(image, encoder, generator, critic, image_shape):
    '''
    Get the f-AnoGAN anomaly scores
    '''
    with torch.no_grad():
        true_img = prepare_input(image, image_shape)
        true_z = encoder(true_img)
        fake_img = generator(true_z.view(1, -1, 1, 1))
        fake_z = encoder(fake_img)
        true_feat = critic.forward_features(true_img)
        fake_feat = critic.forward_features(fake_img)
        imgs_dist = F.mse_loss(true_img, fake_img)
        feat_dist = F.mse_loss(true_feat, fake_feat)
        latn_dist = F.mse_loss(true_z, fake_z) 
        return np.array([imgs_dist.item(), 
                         feat_dist.item(), 
                         latn_dist.item()])


def get_resnet(device='cpu'):
    '''
    Returns a model used only for deep feature extraction,
    tuning only BN parameters and fixing all the conv layers weights.  
    '''
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)    
    resnet.fc = nn.Identity() # type: ignore    
    freeze_module(resnet)
    return resnet.to(device)


def resnet_features(image, resnet, image_shape):
    ''' Get the ResNet50 features '''
    with torch.no_grad():
        img_tensor = prepare_input(image, image_shape).repeat(1, 3, 1, 1)
        return resnet(img_tensor)[0].numpy()
    

def freeze_module(module: nn.Module) -> None:
    ''' 
    Freeze module parameters
    '''
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    ''' 
    Reverse module freezing. 
    '''
    for param in module.parameters():
        param.requires_grad = True 