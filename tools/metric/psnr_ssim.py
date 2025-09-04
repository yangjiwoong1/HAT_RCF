'''
implementation of psnr and ssim with gaussian kernel
refer to https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/metrics/psnr_ssim.py
'''

import numpy as np
import cv2

def calculate_psnr(img, img2, crop_border=4, **kwargs):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img, img2, full=False, win_size=11, sigma=1.5):
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(win_size, sigma)
    window = np.outer(kernel, kernel.transpose())

    pad = win_size // 2
    mu1 = cv2.filter2D(img, -1, window)[pad:-pad, pad:-pad]
    mu2 = cv2.filter2D(img2, -1, window)[pad:-pad, pad:-pad]

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img**2, -1, window)[pad:-pad, pad:-pad] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[pad:-pad, pad:-pad] - mu2_sq
    sigma12 = cv2.filter2D(img*img2, -1, window)[pad:-pad, pad:-pad] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean() if not full else (ssim_map.mean(), ssim_map)

def calculate_ssim(img, img2, win_size=11, sigma=1.5, full=False):

    if full:
        return _ssim(img, img2, win_size=win_size, sigma=sigma, full=full)

    ssims = []
    if len(img.shape) == 2:
        return _ssim(img, img2, win_size=win_size, sigma=sigma)
    else:
        for i in range(img.shape[2]):
            ssims.append(_ssim(img[..., i], img2[..., i], win_size=win_size, sigma=sigma))
    return np.mean(ssims)
