import warnings

import numpy as np
import skimage
from scipy import stats


def percnorm(nii_data, perc=95):
    '''
    Perform percentile-based normalization on the input image.
    '''
    mean, std = nii_data.mean(), nii_data.std()
    z_scores = (nii_data - mean) / std 
    p = (perc + (100 - perc) / 2) / 100 # from 1 tail to 2 tails
    q = stats.norm.ppf(p, loc=0, scale=1)
    z_scores[ z_scores < -q ] = -q
    z_scores[ z_scores >  q ] =  q
    return (z_scores * std) + mean


def brain_cropping(scan, margin, threshold_function=skimage.filters.threshold_otsu):
    '''
    Crop the brain inside the scan, minimizing the amount of black background. 
    The cropping can be amortized using a margin. The standard thresholding function
    is Otsu, but it can be changed with any other skimage.filters.thresholds
    '''
    t = threshold_function(scan)
    binary_image = scan > t
    h, w = binary_image.shape

    if np.all(binary_image == False):
        warnings.warn("Argument scan is totally black.")
        return scan

    for right_to_left in range(0, w):
        if any(binary_image[:, right_to_left]): break

    for left_to_right in range(w-1, -1, -1):
        if any(binary_image[:, left_to_right]): break

    for top_to_bottom in range(0, h):
        if any(binary_image[top_to_bottom, :]): break

    for bottom_to_top in range(h-1, -1, -1):
        if any(binary_image[bottom_to_top, :]): break
            
    r_anchor = right_to_left - margin if right_to_left - margin > 0 else right_to_left # type: ignore
    l_anchor = left_to_right + margin if left_to_right + margin < w else left_to_right # type: ignore
    t_anchor = top_to_bottom - margin if top_to_bottom - margin > 0 else top_to_bottom # type: ignore
    b_anchor = bottom_to_top + margin if bottom_to_top + margin < h else bottom_to_top # type: ignore 

    return scan[t_anchor:(b_anchor+1), r_anchor:(l_anchor+1)]


def square_padding(scan, resize_to=None):
    '''
    Pads the lowest dimension of the image to make it shaped
    like a square. Eventually, the output image can be reshaped
    using the `resize_to` argument.  
    '''
    h, w = scan.shape
    max_dim = max(scan.shape)
    output_shape_placeholder = np.zeros((max_dim, max_dim))
    top_left_x = int(np.round((max_dim - h) / 2))
    top_left_y = int(np.round((max_dim - w) / 2))
    output_shape_placeholder[top_left_x:(top_left_x + h), top_left_y:(top_left_y + w)] = scan.copy()
    if resize_to is not None:
        return skimage.transform.resize(output_shape_placeholder, resize_to)
    return output_shape_placeholder


def normalize_scan(slicearr, img_size=300, margin=1):
    '''
    Applies zscore normalization, resize to IMG_SIZE x IMG_SIZE, and z-score normalization.
    '''
    slicearr_cp = slicearr.copy()
    slicearr_cp = brain_cropping(slicearr_cp, margin=margin)    
    slicearr_cp = square_padding(slicearr_cp, resize_to=(img_size, img_size))
    slicearr_cp = stats.zscore(slicearr_cp, axis=None) # type: ignore
    return slicearr_cp
