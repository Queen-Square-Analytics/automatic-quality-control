import numpy as np
from scipy.stats import norm

from .ksf import k_space_features
from .qcf import image_space_features
from .deep import critic_features, resnet_features
from qcs.fanogan.model import Discriminator


def deterministic_normal_sampling(n_points: int, 
                                  max_value: int, 
                                  min_value: int = 0, 
                                  cut_range: float = 0.4):
    '''
    Suppose we have a range of indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].
    First, we reduce the range around the middle taking only 40% of it, so we will
    consider only [ 3, 4, 5, 6, 7 ] as candidates indices. Then we sample n_points
    in this range deterministically, but following a normal distribution. 
    The resulting indices have higher density around the center. 
    
    The algorithm is well explained here: 
    https://math.stackexchange.com/questions/868755/generating-a-nonrandom-sequence-which-has-a-normal-distributed-density
    '''
    max_value -= 1
    mid = (max_value - min_value) // 2
    halfrange = max_value - mid
    sampling_halfrange = int(halfrange * cut_range) 
    
    zl = mid + sampling_halfrange
    std = (zl - mid) / norm.ppf(1 / (n_points * 2))
    
    points = []
    for i in range(1, n_points + 1):
        p = int(mid + norm.ppf(( 2. * i - 1 ) / (n_points * 2)) * std)
        points.append(p)
    return points


def extract_slices_with_dns(nii_data, n_slices, force_axis=None):
    '''
    Extract n slices from the axis with highest resolution. The sampling is done in 
    a restricted range (40% of the slices before and after the center) to cut out slices
    without information, and it's done following a normal distribution, but deterministically. 
    '''
    max_resolution_axis = int(np.argmin(nii_data.shape)) if force_axis is None else force_axis
    max_resolution_axis_nslices = nii_data.shape[max_resolution_axis]
    if n_slices > 1:
        indices = deterministic_normal_sampling(n_slices, max_resolution_axis_nslices)
    else:
        indices = [ nii_data.shape[max_resolution_axis] // 2 ]
    return [ np.take(nii_data, indices=i, axis=max_resolution_axis) for i in indices ], indices


def extract_features_i(image: np.ndarray, nan_to_num_v=-100) -> np.ndarray:
    ''' Extract imaging-space features '''
    f = image_space_features(image)
    return np.nan_to_num(f, nan=nan_to_num_v, posinf=nan_to_num_v, neginf=nan_to_num_v)
    

def extract_features_k(image: np.ndarray, nan_to_num_v=-100) -> np.ndarray:
    ''' Extract k-space statistical features '''
    f = k_space_features(image)
    return np.nan_to_num(f, nan=nan_to_num_v, posinf=nan_to_num_v, neginf=nan_to_num_v)


def extract_features_g(image: np.ndarray, enc, gen, critic, img_shape) -> np.ndarray:
    ''' Extract anomaly scores (f-AnoGAN)'''
    return critic_features(image, enc, gen, critic, img_shape)


def extract_features_r(image: np.ndarray, resnet, img_shape: tuple) -> np.ndarray:
    ''' Extract deep features from pre-trained ResNet50'''
    return resnet_features(image, resnet, img_shape)
