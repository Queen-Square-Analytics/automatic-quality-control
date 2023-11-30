import random
import numpy as np
from numpy.fft import fft2, fftshift
from skimage.exposure import rescale_intensity

from qcs.artefacts import utils
from qcs.artefacts.base import BaseTransform


class ZipperTransform(BaseTransform):

    def __init__(self, severity=3, params=None):
        super().__init__(severity, params)

    @property
    def default_params(self):
        return np.array([ 7, 20, 20, 30 ])

    @property
    def params_bounds(self):
        return [
            (3,  10), 
            (10, 40),
            (10, 30), 
            (10, 50)
        ]

    def transform(self, img, severity=None):
        super().transform(img, severity)
        h = img.shape[0]

        par1_zipper = self.p[0]  # Maximum number of zipper lines (-1)
        par3_zipper = self.p[1]  # Maximum size of a zipper line
        par4_zipper = random.random() * self.p[3] + self.p[2]  # intensity scaling factor

        n_zippers = 1 + int(self.rng() * par1_zipper) 
        par5_zipper_sp = np.zeros(n_zippers, dtype='int32')
        par5_zipper_ep = np.zeros(n_zippers, dtype='int32')

        for pp in range(n_zippers):
            par5_zipper_sp[pp] = int(random.random() * h)
            par5_zipper_ep[pp] = int(self.rng() * par3_zipper + 1)

        output_image = rescale_intensity(img, out_range=np.uint8)
        kspace = fftshift(fft2(img))

        for pp in range(n_zippers):
            X = (kspace * utils.complex_rand_from_reference(kspace.shape, kspace)) / par4_zipper
            X = X.real.astype(np.uint8)
            s = par5_zipper_sp[pp]
            e = par5_zipper_ep[pp]
            output_image[s:min(s+e,h), :] = X[s:min(s+e, h), :]

        return utils.pad_to_shape(output_image, img.shape)

