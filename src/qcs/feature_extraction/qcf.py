import warnings

import numpy as np
import skimage
from numpy import pad
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d as conv2
from skimage import exposure as ex
from skimage.filters import median, threshold_otsu
from skimage.measure import shannon_entropy
from skimage.morphology import convex_hull_image, square
from skimage.exposure import rescale_intensity

from qcs.artefacts import utils


warnings.filterwarnings("ignore")

SIZE_PATCHES = 21
EPS = 1e-6


def image_space_features(image):
    '''
    '''
    # some errors are casued by negative values 
    # of the image. We can avoid this problem by 
    # normalizing the image in [0, 255] before 
    # computing the metrics
    image = (image - image.min()) / (image.max() - image.min())

    F, B, c, f, b = extract_foreground(image)
    
    
    # F: Image containing the foreground
    # B: Image containing the background
    # c: Binary matrix of the convex hull
    # f: 1-D array containing only pixels belonging to the foreground
    # b: 1-D array containing only pixels belonging to the background
     
    features = [
        mean(f),
        rang(f),
        variance(f),
        percent_coefficient_variation(f),
        contrast_per_pixel(F),
        fpsnr(F),
        snr1(f, b),
        snr2(F, b),
        snr3(F),
        snr4(F, B),
        cnr(F, B),
        cvp(F),
        cjv(f, b),
        efc(F),
        fber(f, b),
        discontinuity1(F),
        discontinuity2(F),
        discontinuity3(F),
        discontinuity4(F),
        discontinuity5(F),
        discontinuity6(F),
        discontinuity7(F),
        discontinuity8(F),
        discontinuity9(F),
        band_detector1(F), # usually invalid
        band_detector2(B), # usually invalid
        band_detector3(B),
        band_detector4(B),
        band_detector5(B),
        band_detector6(B),
        band_detector7(B),
        band_detector8(B) 
    ]       
    return np.array(features)


def extract_foreground(img):
    try:
        h = ex.equalize_hist(img[:, :]) * 255
        oi = np.zeros_like(img, dtype=np.uint16)
        oi[(img > threshold_otsu(img)) == True] = 1
        oh = np.zeros_like(img, dtype=np.uint16)
        oh[(h > threshold_otsu(h)) == True] = 1
        nm = img.shape[0] * img.shape[1]
        w1 = np.sum(oi) / nm
        w2 = np.sum(oh) / nm
        ots = np.zeros_like(img, dtype=np.uint16)
        new = (w1 * img) + (w2 * h)
        ots[(new > threshold_otsu(new)) == True] = 1
        conv_hull = convex_hull_image(ots)
        ch = np.multiply(conv_hull, 1)
        fore_image = ch * img
        back_image = (1 - ch) * img
    except Exception:
        if img is None:
            fore_image = np.ones_like(img, dtype=np.uint16)
        else:
            fore_image = img.copy()
        back_image = np.zeros_like(img, dtype=np.uint16)
        conv_hull = np.zeros_like(img, dtype=np.uint16)

    return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull == False]


def mean(f):
    return np.nanmean(f)


def rang(f):
    try:
        return np.ptp(f)
    except:
        return 0.


def variance(f):
    return np.nanvar(f)


def percent_coefficient_variation(f):
    return (np.nanstd(f) / np.nanmean(f)) * 100


def contrast_per_pixel(F):
    filt = np.array([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])
    I_hat = conv2(F, filt, mode='same')  # type: ignore
    return np.nanmean(I_hat)


def psnr(img1, img2):
    mse = np.square(np.subtract(img1, img2)).mean()
    return 20 * np.log10(np.nanmax(img1) / np.sqrt(mse))


def fpsnr(F):
    I_hat = median(F / np.max(F), square(5))
    return psnr(F, I_hat)


def snr1(f, b):
    return np.nanstd(f) / np.nanstd(b)


def patch(img, patch_size):
    h = int(np.floor(patch_size / 2))
    U = pad(img, pad_width=h, mode='constant')
    [a, b] = np.where(img == np.max(img))
    a = a[0]
    b = b[0]
    return U[a:a + 2 * h + 1, b:b + 2 * h + 1]


def snr2(F, b):
    fore_patch = patch(F, 5)
    return np.nanmean(fore_patch) / np.nanstd(b)


def snr3(F):
    fore_patch = patch(F, 5)
    return np.nanmean(fore_patch) / np.nanstd(fore_patch - np.nanmean(fore_patch))


def snr4(F, B):
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    return np.nanmean(fore_patch) / np.nanstd(back_patch)


def cnr(F, B):
    fore_patch = patch(F, 5)
    back_patch = patch(B, 5)
    return np.nanmean(fore_patch - back_patch) / np.nanstd(back_patch) # type: ignore


def cvp(F):
    fore_patch = patch(F, 5)
    return np.nanstd(fore_patch) / np.nanmean(fore_patch)


def cjv(f, b):
    return (np.nanstd(f) + np.nanstd(b)) / abs(np.nanmean(f) - np.nanmean(b))


def efc(F):
    n_vox = F.shape[0] * F.shape[1]
    efc_max = 1.0 * n_vox * (1.0 / np.sqrt(n_vox)) * \
              np.log(1.0 / np.sqrt(n_vox))
    cc = (F ** 2).sum()
    b_max = np.sqrt(abs(cc))
    return float((1.0 / abs(efc_max)) * np.sum(
        (F / b_max) * np.log((F + 1e16) / b_max)))


def fber(f, b):
    fg_mu = np.nanmedian(np.abs(f) ** 2)
    bg_mu = np.nanmedian(np.abs(b) ** 2)
    if bg_mu < 1.0e-3:
        return 0
    return float(fg_mu / bg_mu)


def edge_detector(F):
    filt = np.array([[-1, -1, -1],
                     [-1, 7, -1],
                     [-1, -1, -1]])

    I_hat = conv2(F, filt, mode='same')
    return I_hat


def edge_detector2(B):
    filt = np.array([[-1, -1, -1],
                     [-1, 10, -1],
                     [-1, -1, -1]])

    I_hat = conv2(B, filt, mode='same')
    I_hat = rescale_intensity(I_hat, out_range=np.uint8)
    return I_hat


def discontinuity1(F):
    I_hat = edge_detector(F)
    result = np.max(np.abs(np.max(I_hat, axis=1)))
    return result


def discontinuity2(F):
    I_hat = edge_detector(F)
    result = np.max(np.abs(np.var(I_hat, axis=1)))
    return result


def discontinuity3(F):
    I_hat = edge_detector(F)
    result = np.max(np.abs(np.convolve(np.var(I_hat, axis=1), [-1, 1], mode='same')))
    return result


def discontinuity4(F):
    I_hat = edge_detector(F)
    result = np.max(np.abs(np.convolve(np.max(I_hat, axis=1), [-1, 1], mode='same')))
    return result


def discontinuity5(F):
    result = np.sum(np.abs(F[5:-5]), axis=1)
    return np.min(result)


def discontinuity6(F):
    result = np.sum(np.abs(F[5:-5]), axis=1)
    return np.max(result)


def discontinuity7(F):
    result = np.sum(np.abs(F[5:-5]), axis=0)
    return np.min(result)


def discontinuity8(F):
    result = np.sum(np.abs(F[5:-5]), axis=0)
    return np.max(result)


def discontinuity9(F):
    curr_array = np.max(F, axis=1)
    result = np.zeros(np.shape(curr_array)[0])
    for i in range(1, np.shape(curr_array)[0] - 1):
        if curr_array[i - 1] > 100 and curr_array[i] == 0 and curr_array[i + 1] > 100:
            result[i] = 1
    return np.sum(result)


def band_detector1(F):
    return compute_global_contrast_factor(F)


def band_detector2(B):
    return compute_global_contrast_factor(B)


def band_detector3(B):
    return stats.iqr(B)


def band_detector4(B):
    patches = extract_4_patches(B)
    result = np.median([stats.iqr(patches[0, :, :]), stats.iqr(patches[1, :, :]), stats.iqr(patches[2, :, :]), stats.iqr(patches[3, :, :])])
    return result


def band_detector5(B):
    patches = extract_4_patches(B)
    result = np.median(
        [shannon_entropy(patches[0, :, :]), shannon_entropy(patches[1, :, :]), shannon_entropy(patches[2, :, :]), shannon_entropy(patches[3, :, :])]) # type: ignore
    return result


def band_detector6(B):
    I_hat = edge_detector2(B)
    return np.mean(I_hat)


def band_detector7(B):
    I_hat = edge_detector2(B)
    return np.var(I_hat)


def band_detector8(B):
    I_hat = edge_detector2(B)
    return shannon_entropy(I_hat)


def extract_4_patches(B):
    h = B.shape[0]
    w = B.shape[1]
    B = gaussian_filter(B, sigma=0.2)
    B = edge_detector2(B)
    patches = np.zeros((4, SIZE_PATCHES, SIZE_PATCHES))
    patches[0, :, :] = B[:SIZE_PATCHES, :SIZE_PATCHES]
    patches[1, :, :] = B[h - SIZE_PATCHES:h, :SIZE_PATCHES]
    patches[2, :, :] = B[:SIZE_PATCHES, w - SIZE_PATCHES:w]
    patches[3, :, :] = B[h - SIZE_PATCHES:h, w - SIZE_PATCHES:w]
    return patches


def compute_global_contrast_factor(img):
    initial_size = np.shape(img)[0]
    gr = np.float32(img)
    superpixel_sizes = [1, 2, 4, 8, 16, 25, 50]
    gcf = 0

    for i, size in enumerate(superpixel_sizes, 1):
        wi = (-0.406385 * i / 9 + 0.334573) * i / 9 + 0.0877526
        im_scale = skimage.transform.resize(gr, (int(initial_size/size), int(initial_size/size)), anti_aliasing=False)
        avg_contrast_scale = compute_image_average_contrast(im_scale)
        gcf += wi * avg_contrast_scale
    return gcf


def compute_image_average_contrast(k, gamma=2.2):
    L = 100 * np.sqrt((np.float32(k) / 255) ** gamma)
    # pad image with border replicating edge values
    L_pad = np.pad(L, 1, mode='edge')

    # compute differences in all directions
    left_diff   = L - L_pad[1:-1, :-2]
    right_diff  = L - L_pad[1:-1, 2:]
    up_diff     = L - L_pad[:-2, 1:-1]
    down_diff   = L - L_pad[2:, 1:-1]

    # create matrix with number of valid values 2 in corners, 3 along edges and 4 in the center
    num_valid_vals = 3 * np.ones_like(L)
    num_valid_vals[[0, 0, -1, -1], [0, -1, 0, -1]] = 2
    num_valid_vals[1:-1, 1:-1] = 4
    pixel_avgs = (np.abs(left_diff) + np.abs(right_diff) + np.abs(up_diff) + np.abs(down_diff)) / num_valid_vals
    return np.mean(pixel_avgs)