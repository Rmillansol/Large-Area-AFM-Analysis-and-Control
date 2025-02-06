"""
Author: Ruben Millan-Solsona
Date of Creation: August 2024

Description:
This module contains functions for flattening images, including methods to reduce
noise in images, particularly for atomic force microscopy (AFM) data. It offers plane 
and polynomial surface fitting, optimal plane subtraction, and various filtering techniques
to improve image quality.

Dependencies:
- os
- numpy
- cv2
- matplotlib.pyplot
- typing
- scipy.optimize
- scipy.ndimage
- scipy.stats
- skimage.metrics
- skvideo.measure
- AFMclasses (contains clImage, ChannelType, ExtentionType)
- managefiles (custom module for file management)

"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.optimize import curve_fit, differential_evolution
from scipy.ndimage import label, find_objects
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from AFMclasses import clImage, ChannelType, ExtentionType
import managefiles as mgf

def SaveMatriz(img, fileName, path, imgPro=None):
    """
    Saves a matrix in WSxM ASCII XYZ format based on properties of imgPro.
    """
    dir_path = os.path.join(path, fileName)
    if imgPro is None:
        step_x = img.shape[0]
        step_y = img.shape[1]
        unitxy = 'um'
        unitz = 'nm'
    else:
        step_x = imgPro.size_x / img.shape[0]
        step_y = imgPro.size_y / img.shape[1]
        unitxy = imgPro.unitxy
        unitz = imgPro.unitz

    with open(dir_path, 'w') as file:
        file.write("WSxM file copyright UAM\n")
        file.write("WSxM ASCII XYZ file\n")
        file.write(f"X[{unitxy}]\tY[{unitxy}]\tZ[{unitz}]\n")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                x_real = i * step_x
                y_real = j * step_y
                file.write(f"{x_real} {y_real} {img[i, j]}\n")

def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Resizes the image to the specified size using OpenCV.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def plane(coords, a, b, c):
    """
    Defines a plane function: ax + by + c.
    """
    x, y = coords
    return a * x + b * y + c

def poly(coords, a, b, c, d, e, f):
    """
    Defines a polynomial surface function: ax^2 + by^2 + cxy + dx + ey + f.
    """
    x, y = coords
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

def FitPoly(img, mask=None):
    """
    Fits a polynomial surface to the image data, optionally applying a mask to focus on certain regions.
    Returns the fitted surface and the Pearson correlation.
    """
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)

    if mask is None:
        x_flat, y_flat, z_flat = x.flatten(), y.flatten(), img.flatten()
    else:
        x_flat, y_flat, z_flat = x[mask == 1].flatten(), y[mask == 1].flatten(), img[mask == 1].flatten()
    
    p0 = np.zeros(6)
    params, _ = curve_fit(poly, (x_flat, y_flat), z_flat, p0)
    poly_fitted = poly((x, y), *params).reshape(img.shape)
    correlation, _ = pearsonr(z_flat, poly((x_flat, y_flat), *params))

    return poly_fitted, correlation

def SubtractGlobalPoly(img, show=False):
    """
    Subtracts a polynomial surface from the image and optionally displays the result.
    """
    poly_fitted, correlation = FitPoly(img)
    img_flattened = img - poly_fitted

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(img_flattened, cmap='gray')
        ax[1].set_title('Flattened Image')
        plt.show()

    return img_flattened, poly_fitted

def SubtractGlobalPlane(img, show=False):
    """
    Subtracts a fitted plane from the image and optionally displays the result.
    """
    plane_fitted, correlation = FitPlane(img)
    img_flattened = img - plane_fitted

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].imshow(img_flattened, cmap='gray')
        ax[1].set_title('Flattened Image')
        plt.show()

    return img_flattened, plane_fitted

def FitOffsetToFlattingImageByDiffAndMask(img, mask=None):
    """
    Calculates the offset between adjacent lines in an image using a mask to ignore certain pixels.
    """
    offset_img = img.copy()
    for i in range(1, offset_img.shape[0]):
        line_below, current_line = offset_img[i - 1, :], offset_img[i, :]
        if mask is not None:
            line_below_masked, current_line_masked = line_below[mask[i, :] == 1], current_line[mask[i, :] == 1]
            offset = np.median(current_line_masked - line_below_masked) if len(line_below_masked) > 0 else 0
        else:
            offset = np.median(current_line - line_below)
        offset_img[i, :] -= offset

    if mask is not None:
        offset_img[mask == 0] = 0

    return offset_img

def equalize_median_with_std_derivative_threshold(img, factor):
    """
    Adjusts each row of the image so that the median of the current line matches the previous line.
    Points with a high derivative, based on a threshold from the standard deviation, are ignored.
    """
    adjusted_img = img.copy()
    for i in range(1, adjusted_img.shape[0]):
        current_line, previous_line = adjusted_img[i, :], adjusted_img[i - 1, :]
        derivative = np.abs(np.diff(current_line))
        std_derivative = np.std(derivative)
        threshold = factor * std_derivative
        valid_mask = np.ones_like(current_line, dtype=bool)
        valid_mask[1:][derivative > threshold] = False
        current_median = np.median(current_line[valid_mask])
        previous_median = np.median(previous_line)
        offset = previous_median - current_median
        adjusted_img[i, :] += offset

    return adjusted_img

# ************************** Auto Functions **************************

def AutoFlattenPlus(img, perc=0.4):
    """
    Automatically flattens the image using differential offset, optimal plane, and FlattenPlus methods.
    """
    imgDiff = FitOffsetToFlattingImageByDiff(img)
    image_corrected, fitted_plane = SubtractOptimalPlane(imgDiff, 0.1, show=False)
    img_plus, mask = FlattenPlus(image_corrected, perc)
    return img_plus, mask

# Display helper function
def display_images(img1, img2, img_combined, titles):
    """
    Displays three images side by side with color scales and titles.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(titles[0])
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(titles[1])
    fig.colorbar(im2, ax=axes[1])
    im3 = axes[2].imshow(img_combined, cmap='gray')
    axes[2].set_title(titles[2])
    fig.colorbar(im3, ax=axes[2])
    plt.show()

if __name__ == '__main__':
    pth = r'C:\Users\z78\Documents\Captures\DriveAFM\2024\Oct\21th'
    fn = 'Adama_Pa_2_16.gwy'
    img = mgf.LoadChannelGWY_FileToImage(filename=fn, path=pth)
    image_corrected, mask = SubtractGlobalPlane(img.matriz, show=True)
    SaveMatriz(image_corrected, 'TestImage_der.txt', '', imgPro=None)
