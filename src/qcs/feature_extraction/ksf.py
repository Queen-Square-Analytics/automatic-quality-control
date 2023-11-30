import math
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

TOT_FEATURES = 50 # 12 stats x 4 func + 2 single values


def k_space_features(img):
    kspace, low_kspace, high_kspace = k_space(img)
    
    features = [
        *norm_hist_stats(kspace), 
        *norm_hist_stats(low_kspace), 
        *norm_hist_stats(high_kspace), 
        *K4(img, kspace),
        blurred_features(img, kspace),
        find_total_pick(low_kspace, high_kspace),
    ]
    return np.array(features)


def k_space(img, size=60):
    (h, w) = img.shape
    fft = np.fft.fft2(img)
    all_k_Space = np.fft.fftshift(fft)

    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    low_freq = all_k_Space[cY - size:cY + size, cX - size:cX + size]
    high_freq = all_k_Space.copy()
    high_freq[cY - size:cY + size, cX - size:cX + size] = 0
    return all_k_Space, low_freq, high_freq


def extract_stat_features(m):
    return [
        *stats.describe(m)[2:],
        stats.kstat(m), 
        stats.kstatvar(m), 
        stats.tstd(m),
        stats.tsem(m), 
        stats.variation(m), 
        stats.iqr(m),
        stats.sem(m), 
        stats.entropy(m)
    ]


def norm_hist_stats(kspace):
    return extract_stat_features(compute_normalized_hist(kspace.copy()))


def compute_normalized_hist(image):
    magnitude = np.abs(image)
    vect_mag = magnitude.reshape(-1)
    bins = np.linspace(min(vect_mag), max(vect_mag), 100)
    weightsa = np.ones_like(vect_mag) / float(len(vect_mag))
    val = np.histogram(magnitude.reshape(-1), bins, weightsa)
    normalized_hist = val[0] / max(val[0] + 0.0000001)
    return normalized_hist


def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X - x + 1), (Y - y + 1), x, y)  # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one
    strides = img.itemsize * np.array([X, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def find_total_pick(L, H):
    return np.max([
        find_pick_H(H.copy()),
        find_pick_L(L.copy())
    ])
    
def find_pick_H(H) -> float:
    return find_pick(H, 7, 5)

def find_pick_L(L) -> float:
    return find_pick(L, 5, 3) 
    
    
# TODO: inspect the linter error on the type: ignore

def find_pick(image, kernel_size, strip) -> float:
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    magnitude = np.abs(image)
    magnitude[cY - 1:cY + 2, :] = 0
    magnitude[:, cX - 1:cX + 2] = 0
    all_patch = patchify(magnitude, [kernel_size, kernel_size])
    all_r = np.zeros((np.shape(all_patch)[0], np.shape(all_patch)[1]))
    for i in range(1, np.shape(all_patch)[0] - 1, strip):
        for j in range(1, np.shape(all_patch)[1] - 1, strip):
            if not (np.median(all_patch[i, j, :, :]) == 0):
                all_r[i, j] = (np.max(all_patch[i, j, :, :]) / np.median(all_patch[i, j, :, :])) * (np.abs(cX - i) + np.abs(cY - j) / 100)
    if np.size(all_r > 0)>0 and np.median(all_r[all_r > 0])>0:
        return np.max(all_r[all_r > 0]) / np.median(all_r[all_r > 0]) # type: ignore
    else:
        return 0.


def K4(image, kspace):
    h, w = image.shape
    kspace_cp = kspace.copy()   
    # changing np.log by adding 1 and preventing -inf and NaN
    magnitude = 20 * np.log(1 + np.abs(kspace_cp))
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    d = np.zeros([int(min(h, w) / 2)])
    for radius in range(0, int(min(h, w) / 2)):
        currSum = 0
        for interval in np.arange(0, 2 * math.pi, 0.1):
            X = cX + (radius * math.cos(interval))
            Y = cY + (radius * math.sin(interval))
            currSum += magnitude[int(np.floor(Y)), int(np.floor(X))]
        
        # changing np.log by adding 1 and preventing -inf
        d[radius] = np.log(1 + currSum)
    return extract_stat_features(d)


def blurred_features(image, _fft_shift):
    fft_shift = _fft_shift.copy()
    size = 60
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean