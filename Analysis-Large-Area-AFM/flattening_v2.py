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
from scipy.optimize import curve_fit, minimize, differential_evolution, basinhopping
from scipy.ndimage import label, find_objects, generic_filter, laplace
from scipy.stats import pearsonr, entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import skvideo.measure
# This function goes out of this module but for now I leave it here
# Save in xyz.txt format
from AFMclasses import clImage, ChannelType, ExtentionType
import managefiles as mgf
# Function that saves a matrix in WXSM format according to IMGPRO properties
def SaveMatriz(img, fileName, path, imgPro = None):
    dir_path = os.path.join(path, fileName)
    
    if imgPro is None:
        step_x = img.shape[0]
        step_y = img.shape[1]
        unitxy = 'um'
        unitz = 'nm'
    else:
        # We calculate the steps on the x and y axes
        step_x = imgPro.size_x / img.shape[0]  # Total size in x divided by the number of pixels in x
        step_y = imgPro.size_y / img.shape[1]  # Total size and divided by the number of pixels in and
        unitxy = imgPro.unixy
        unitz = imgPro.uniz

    with open(dir_path, 'w') as file:
        file.write("WSxM file copyright UAM\n")
        file.write("WSxM ASCII XYZ file\n")
        file.write(f"X[{unitxy}]\tY[{unitxy}]\tZ[{unitz}]\n")
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # We convert the indices to physical coordinates
                x_real = i * step_x
                y_real = j * step_y
                file.write(f"{x_real} {y_real} {img[i, j]}\n")

# Function that resizes a matrix in a smaller or larger
def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize the image to the desired size using OpenCV."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# Function to define the Ax + By + C plane
def plane(coords, a, b, c):
    x, y = coords
    return a * x + b * y + c

# Ax2 +By2 +Cxy +D surface function
def poly(coords, a,b,c,d,e,f):
    x, y = coords
    return a * x**2 + b * y**2 + c*x*y + d*x +e*y+f

# Function that adjusts a polysuperficie
def FitPoly(img, mask=None):
    # Create x and y coordinates
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)

    if mask is None:
        # Flatten data for adjustment
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = img.flatten()
    else:
        # Apply the mask to select only the desired regions
        x_flat = x[mask == 1].flatten()
        y_flat = y[mask == 1].flatten()
        z_flat = img[mask == 1].flatten()
    
    # Initial estimate of the parameters
    p0 = np.array([0, 0, 0, 0, 0, 0])  # Adjust according to the problem

    # Plane adjustment
    params, _ = curve_fit(poly, (x_flat, y_flat), z_flat,p0)

    # Create the adjusted plane
    poly_fitted = poly((x, y), *params).reshape(img.shape)

    # Calculate Pearson's correlation coefficient
    z_fitted_masked = poly((x_flat, y_flat), *params)
    correlation, _ = pearsonr(z_flat, z_fitted_masked)

    return poly_fitted, correlation

# Function that adjusts a plane according to a mask
def FitPlane(img, mask=None):
    # Create x and y coordinates
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)

    if mask is None:
        # Flatten data for adjustment
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = img.flatten()
    else:
        # Apply the mask to select only the desired regions
        x_flat = x[mask == 1].flatten()
        y_flat = y[mask == 1].flatten()
        z_flat = img[mask == 1].flatten()

    # Plane adjustment
    params, _ = curve_fit(plane, (x_flat, y_flat), z_flat)

    # Create the adjusted plane
    plane_fitted = plane((x, y), *params).reshape(img.shape)

    # Calculate Pearson's correlation coefficient
    z_fitted_masked = plane((x_flat, y_flat), *params)
    correlation, _ = pearsonr(z_flat, z_fitted_masked)

    return plane_fitted, correlation

# Function that subtracts a poly from an image
def SubtractGlobalPoly(img, show=False):
    # Create the adjusted plane
    poly_fitted, correlation = FitPoly(img)

    # Subtract the adjusted plane from the original image
    img_flattened = img - poly_fitted

    if show:
        # Show the original image and the flattened image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        cax0 = ax[0].imshow(img, cmap='gray', interpolation='none', origin='upper')
        ax[0].set_title('original image')
        fig.colorbar(cax0, ax=ax[0], orientation='vertical')

        cax1 = ax[1].imshow(img_flattened, cmap='gray', interpolation='none', origin='upper')
        ax[1].set_title('flattened image')
        fig.colorbar(cax1, ax=ax[1], orientation='vertical')

        plt.show()

    return img_flattened, poly_fitted

# Function that subtracts the global plane
def SubtractGlobalPlane(img, show=False):
    # Create the adjusted plane
    plane_fitted, correlation = FitPlane(img)

    # Subtract the adjusted plane from the original image
    img_flattened = img - plane_fitted

    if show:
        # Show the original image and the flattened image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        cax0 = ax[0].imshow(img, cmap='gray', interpolation='none', origin='upper')
        ax[0].set_title('original image')
        fig.colorbar(cax0, ax=ax[0], orientation='vertical')

        cax1 = ax[1].imshow(img_flattened, cmap='gray', interpolation='none', origin='upper')
        ax[1].set_title('flattened image')
        fig.colorbar(cax1, ax=ax[1], orientation='vertical')

        plt.show()

    return img_flattened, plane_fitted

def SubtracGlobalPlaneManualByPoints(imagen, show=False):
    def seleccionar_puntos(event, puntos, mask):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            puntos.append((x, y))
            mask[y, x] = 1  # Mark the point in the mask
            if len(puntos) <= 100:  # Show up to 10 points to give reference
                plt.plot(x, y, 'ro')
                plt.draw()
   # Initialize selected points mask
    mask = np.zeros_like(imagen, dtype=int)

    # Show image and allow points selection
    fig, ax = plt.subplots()
    ax.imshow(imagen, cmap='gray')
    puntos = []
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: seleccionar_puntos(event, puntos, mask))
    plt.title("Seleccione puntos (clic para agregar)")
    plt.show()

    fig.canvas.mpl_disconnect(cid)

    if len(puntos) < 3:
        print("Se necesitan al menos tres puntos para calcular un plano.")
        return mask  # The mask returns even if it has few points
    # Adjust plane by mask
    img_flattened, plane_fitted = SubtractPlanebyMask(imagen,mask,show=show)

    return img_flattened,mask
# Function that subtracts the optimal plane that maximizes the STD and puts a percentage of the lowest to zero medium
def SubtractOptimalPlane(image, perc = 0.1, show=False):
    
    def adjust_lowest_median_to_zero(image, fitted_plane, percentage_lowest):
        
        # Subtract the adjusted plane from the image
        image_corrected = image - fitted_plane

        # Flatten the corrected image to easily operate with values
        image_flat = image_corrected.flatten()

        # Order the corrected image values ​​and select the lowest percentage
        sorted_indices = np.argsort(image_flat)
        n_points = int(len(image_flat) * percentage_lowest)
        lowest_indices = sorted_indices[:n_points]

        # Obtain the values ​​corresponding to the lowest indices
        lowest_values = image_flat[lowest_indices]
    
        # Calculate the average of the lowest values
        median_lowest = np.median(lowest_values)
        
        # Adjust the image so that the lowest values ​​have zero medium
        image_corrected -= median_lowest

        return image_corrected
    
    # Code use:
    fitted_plane, result = find_optimal_plane(image)
    image_corrected = adjust_lowest_median_to_zero(image, fitted_plane, perc)
    # Subtract the adjusted plane from the image
    # Simage_Corrected = image - fitted_plane
    if show:
        # Now we show the three images: original, adjusted plane, and the corrected image
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(image, cmap='viridis')
        axes[0].set_title('Imagen Original de Altura')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')

        # Adjusted plane
        axes[1].imshow(fitted_plane, cmap='viridis')
        axes[1].set_title('Plano Ajustado')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')

        # Corrected image
        im = axes[2].imshow(image_corrected, cmap='viridis')
        axes[2].set_title('Imagen Corregida (Original - Plano)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')

        # We add a colored bar for the corrected image
        fig.colorbar(im, ax=axes[2])

        plt.tight_layout()
        plt.show()
    return image_corrected, fitted_plane
    
# Function that generates a mask according to a threshold and a minimum area
def MakeMaskWithThreshold(img, umbral,area_minima=10,show=0):

    # Create a mask where pixels are less than or equal to the threshold
    img_threshold = (img < umbral).astype(np.uint8)
    
    # Label connected components
    labeled_array, num_features = label(img_threshold )

    # Create a mask for the regions that meet the minimum area
    mask = np.zeros_like(img_threshold )

    # Iterate on each component labeling
    for i in range(1, num_features + 1):
        slice_x, slice_y = find_objects(labeled_array == i)[0]
        region = labeled_array[slice_x, slice_y] == i
        area = region.sum()
        if area >= area_minima:
        # print (f'area: {region.sum ()} ')
            mask[slice_x, slice_y][region] = 1
            
    # I show mask without Area and with Area
    if show:       
        # Show the Threshold and Areas mask
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Original image
        img1 = ax[0].imshow(img_threshold, cmap='gray')
        ax[0].set_title('mask threshold')
        fig.colorbar(img1, ax=ax[0], orientation='vertical')

        # Image with area filter
        img2 = ax[1].imshow(mask, cmap='gray')
        ax[1].set_title('mask threshold with filter area')
        fig.colorbar(img2, ax=ax[1], orientation='vertical')
        plt.show()

    return mask    

# Function to obtain a mask with a percentage below a certain threshold
def MakeMaskWithPercentage(img, perc=0.1,area_minima=10,show=0):

    # Flatten the pixels array and order
    pixeles_ordenados = np.sort(img.flatten())
    # Calculate the index that corresponds to 10%
    indice_umbral = int(len(pixeles_ordenados) * perc)
    # Get the threshold value
    umbral = pixeles_ordenados[indice_umbral]
    # Threshold mask
    mask = MakeMaskWithThreshold(img, umbral,area_minima,show)
    
    return mask    

# The same as the previous one but uses the mask also to discard points that do not use to calculate
def FitOffsetToFlattingImageByDiffAndMask(img, mask=None):
    """
    Function to calculate the offset based on the weighted average of the differences 
    with adjacent lines, using only the points where the mask equals 1.
    """
    offset_img = img.copy()

    # Apply the offset to the lines
    for i in range(1, offset_img.shape[0]):
        line_below = offset_img[i - 1, :]
        current_line = offset_img[i, :]
        
        # If there is a mask, calculate the offset only at the points where mask == 1
        if mask is not None:
            line_below_masked = line_below[mask[i, :] == 1]
            current_line_masked = current_line[mask[i, :] == 1]
            if len(line_below_masked) > 0 and len(current_line_masked) > 0:  # Avoid empty arrays
                offset = np.median(current_line_masked - line_below_masked)
            else:
                offset = 0  # If there are no points in the mask, do not apply offset
        else:
            # If there is no mask, apply the offset at all points
            offset = np.median(current_line - line_below)

        offset_img[i, :] -= offset

    # Adjust the final offset so that the mask values ​​equal to 1 have zero average
    if mask is not None:
        # offset_img -= np.mean (offset_img [mask == 1])
        # Apply the mask to keep the points where mask == 0
        offset_img[mask == 0] = 0

    return offset_img

def FitLineToFlattingImageByDiffAndMask(img, mask=None):
    """
   Function to calculate the fit of a straight line to each horizontal line
    in the image, using only the points where the mask equals 1.
    """
    def linear_fit(x, a, b):
        return a * x + b

    offset_img = img.copy()

    # Apply the adjustment to the lines
    for i in range(1, offset_img.shape[0]):
        line_below = offset_img[i - 1, :]
        current_line = offset_img[i, :]

        # If there is a mask, calculate the adjustment only to the points where mask == 1
        if mask is not None:
            line_below_masked = line_below[mask[i, :] == 1]
            current_line_masked = current_line[mask[i, :] == 1]
            x_values = np.arange(len(current_line))[mask[i, :] == 1]  # X for points in the mask

            # Adjust a straight line if there are enough points
            if len(line_below_masked) > 2 and len(current_line_masked) > 2:
                # Adjust a straight line to the selected points
                popt, _ = curve_fit(linear_fit, x_values, current_line_masked - line_below_masked, p0=(0, 0))
                fitted_line = linear_fit(np.arange(len(current_line)), *popt)
            else:
                fitted_line = np.zeros_like(current_line)  # If there are no points, do not apply any adjustment
        else:
            # If there is no mask, use all points for adjustment
            x_values = np.arange(len(current_line))
            popt, _ = curve_fit(linear_fit, x_values, current_line - line_below, p0=(0, 0))
            fitted_line = linear_fit(np.arange(len(current_line)), *popt)

        # Subtract the line adjusted to the current line
        offset_img[i, :] -= fitted_line

    # Adjust the final offset so that the mask values ​​equal to 1 have zero average
    if mask is not None:
        offset_img -= np.mean(offset_img[mask == 1])

        # Apply the mask to keep the points where mask == 0
        offset_img[mask == 0] = 0

    return offset_img

# Function that adjusts the plane from a mask
def SubtractPlanebyMask(img,mask,show=0):
     # Create the adjusted plane
    plane_fitted, correlation = FitPlane(img,mask)

    # Subtract the adjusted plane from the original image
    img_flattened = img - plane_fitted

    if show:
        # Show the original image and the flattened image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        cax0 = ax[0].imshow(img, cmap='gray', interpolation='none', origin='upper')
        ax[0].set_title('original image')
        fig.colorbar(cax0, ax=ax[0], orientation='vertical')

        cax1 = ax[1].imshow(img_flattened, cmap='gray', interpolation='none', origin='upper')
        ax[1].set_title('flattened image')
        fig.colorbar(cax1, ax=ax[1], orientation='vertical')

        plt.show()

    return img_flattened, plane_fitted

# function that shows an image of an image according to a mask
def SubtractPolybyMask(img,mask,show=0):
     # Create the adjusted plane
    poly_fitted, correlation = FitPoly(img,mask)

    # Subtract the adjusted plane from the original image
    img_flattened = img - poly_fitted

    if show:
        # Show the original image and the flattened image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        cax0 = ax[0].imshow(img, cmap='gray', interpolation='none', origin='upper')
        ax[0].set_title('original image')
        fig.colorbar(cax0, ax=ax[0], orientation='vertical')

        cax1 = ax[1].imshow(img_flattened, cmap='gray', interpolation='none', origin='upper')
        ax[1].set_title('flattened image')
        fig.colorbar(cax1, ax=ax[1], orientation='vertical')

        plt.show()

    return img_flattened, poly_fitted

# Function that adjusts an offset by Mean to each line of an image according to a mask that indicates flat substrate.
def FitOffsetToFlattingImage(img,mask = None):
    # Adjust the lines that contain zero substrate
    offset_img = img.copy()

    # Apply the offset to the lines
    for i in range(0, offset_img.shape[0]-1):
       
        line_above = offset_img[i + 1, :]
                    
        offset = np.mean(line_above)
        offset_img[i, :] -= offset

    if mask is not None:
        # Displace zero to be the substrate
        offset_img -= np.mean(offset_img[mask == 1])

    
    return offset_img

# Function that adjusts an offset based on the weighted average of the differences with the lines adjacent to each line of an image
def FitOffsetToFlattingImageByDiff(img, mask = None):
    # Function to calculate the offset based on the weighted mean of the differences with the adjacent lines
    offset_img = img.copy()

    # Apply the offset to the lines
    for i in range(1, offset_img.shape[0]):
        line_below = offset_img[i - 1, :]
        current_line = offset_img[i, :]
        offset = np.median(current_line - line_below)
        offset_img[i, :] -= offset

    if mask is not None:
        # Displace zero to be the substrate
        offset_img -= np.mean(offset_img[mask == 1])

    return offset_img

# Function that finds the optimal plane that maximizes the std of the image except the plane
# It readjusts the size when the image is very large but then returns the same size as the urinal
def find_optimal_plane(img1):
    def PlaneWithOutOffset(params, x, y):
        a, b, c = params
        return a * x + b * y +c
    
    def fun(params, x, y, z):
        z_pred = PlaneWithOutOffset(params, x, y)
        return np.std((z - z_pred)**4)

    # Original image dimensions
    M, N = img1.shape

    # We verify if we need to resize the image
    if M > 500:
        imgo = img1  # We save the original image
        # We resize the image at 500x500
        img1 = resize_image(img1, (500, 500))
        M_small, N_small = img1.shape
    else:
        imgo = img1
        M_small, N_small = M, N  # The image is not resolved if it is small

    # We generate the normalized coordinates x and y from the small image
    x_small_normalized, y_small_normalized = np.meshgrid(
        np.linspace(0, 1, N_small), np.linspace(0, 1, M_small)
    )

    # We flatten the matrices x, y y z_data to be one -dimensional vectors
    x_flat = x_small_normalized.ravel()  # Normalized X coordinates
    y_flat = y_small_normalized.ravel()  # Coordinates and normalized
    z_flat = img1.ravel()  # Z values ​​(intensities or heights) of the resized image
    
    
    # We define the limits for the parameters of A and B
    bounds = [(-1e-5, 1e-5), (-1e-5, 1e-5),(-1e-5, 1e-5)]  # Parameter limits
    
    # We use differential_evolution with several adjusted parameters
    result = differential_evolution(
        fun, 
        bounds, 
        args=(x_flat, y_flat, z_flat), 
        seed=42,                 # Set the seed for consistent results
        maxiter=4000,            # Increase the number of iterations
        popsize=25,              # Increase population size
        mutation=(0.5, 1),       # Adjust the mutation factor
    )
    
    # print ("Differential Evolution:")
    # print (f "parameters: {result.x}")
    # print (f "std minimal achieved: {results.fun}")

    # We extract the optimal parameters
    a_opt, b_opt, c_opt = result.x

    # We generate the normalized coordinates x and y for the original image
    x_normalized, y_normalized = np.meshgrid(
        np.linspace(0, 1, N), np.linspace(0, 1, M)
    )

    # We calculate the adjusted plane for the original image using the standardized coordinates
    z_plane = PlaneWithOutOffset([a_opt, b_opt,  c_opt], x_normalized, y_normalized)

    # # Calculate the offset between the original image and the adjusted plane
    # offset = np.mean (imgo - z_plane)

    # # We show the results
    # print (f "optimal plane parameters: a = {a_opt}, b = {b_op}, offset = {offset}")
    
    return z_plane, result

# Function that is missing plus similar to WxSM
# adjust a line to each line taking a percentage of the points of line I
# smaller
def FlattenPlus(img,perc = 40):
    # works very well for samples with a lot of substrate
    def linear_fit(x, a, b):
        return a * x + b

    Flatten_img = img.copy()
    mask = np.zeros_like(img, dtype=bool)  # Create the mask with the same size as the image

    # Apply the adjustment to the lines
    for i in range(0, Flatten_img.shape[0]):
        # Select the points of the current line
        line_current = Flatten_img[i, :]

        # Order the points of the current line by value
        sorted_indices = np.argsort(line_current)
        n_points = int(len(line_current) * perc)

        # Select the percentage of smaller points
        selected_indices = sorted_indices[:n_points]
         # Mark the selected points in the mask
        mask[i, selected_indices] = True
        # Obtain the x and y coordinates of those points
        x_values = np.arange(len(line_current))[selected_indices]
        y_values = line_current[selected_indices]

        # Adjust a straight line to the selected points
        popt, _ = curve_fit(linear_fit, x_values, y_values,p0 = (0,0))

        # Calculate the values ​​of the tight line for the entire row
        fitted_line = linear_fit(np.arange(len(line_current)), *popt)

        # Subtract the line adjusted to the current line
        Flatten_img[i, :] = line_current - fitted_line

        # Adjust the median of the selected points to zero
        median_selected = np.median(Flatten_img[i, selected_indices])
        Flatten_img[i, :] -= median_selected

    return Flatten_img, mask


def FlattenPlus_Mask(img, mask=None, perc=40):
    """
    Fits a line to each horizontal line in the image, using only the lowest points,
    and discarding the points on the mask.
    """
    def linear_fit(x, a, b):
        return a * x + b

    Flatten_img = img.copy()

    # Create a mask if a
    if mask is None:
        mask = np.zeros_like(img, dtype=bool)  # Create the mask with the same size as the image

    # Apply the adjustment to the lines
    for i in range(0, Flatten_img.shape[0]):
        # Select the points of the current line
        line_current = Flatten_img[i, :]

        # Filter the points that are not in the mask
        masked_indices = np.where(mask[i, :])[0]
        line_filtered = line_current[masked_indices]
        # Verify if there are enough points to apply the adjustment
        if len(line_filtered) == 0:
            continue  # Jump this line if there are no points in the mask
        # Order the leaked points by value
        sorted_indices = np.argsort(line_filtered)
        n_points = int(len(line_filtered) * perc / 100)  # Calculate the number of points according to the percentage
        # Verify if there are enough points after applying the percentage
        if n_points < 2:
            continue  # Jump this line if there are not enough points for adjustment

        # Select the percentage of smaller points of the non -masked points
        selected_indices = masked_indices[sorted_indices[:n_points]]

        # Obtain the x and y coordinates of those points
        x_values = np.arange(len(line_current))[selected_indices]
        y_values = line_current[selected_indices]

        # Adjust a straight line to the selected points
        popt, _ = curve_fit(linear_fit, x_values, y_values, p0=(0, 0))

        # Calculate the values ​​of the tight line for the entire row
        fitted_line = linear_fit(np.arange(len(line_current)), *popt)

        # Subtract the line adjusted to the current line
        Flatten_img[i, :] = line_current - fitted_line

        # Apply the mask to keep the points where mask == 0
    Flatten_img[mask == 0] = 0

    return Flatten_img, mask

def FlattenPlus_filtered_v2(img):
    Flatten_img = img.copy()
    mask_selected = np.zeros_like(img, dtype=bool)  # Initialize the selection mask

    # Apply the adjustment to each row of the image
    for i in range(0, Flatten_img.shape[0]):
        # Select the points of the current line
        x_values = np.arange(Flatten_img.shape[1])
        y_values = Flatten_img[i, :]

        # Define the objective function that minimizes the standard deviation of square errors
        def objective(params):
            m, b = params
            fitted_line = m * x_values + b
            squared_errors = (y_values - fitted_line) ** 2
            std_squared_errors = np.std(squared_errors)
            return std_squared_errors

        # Define the limits for the 'm' and 'b' parameters
        bounds = [(-1e-5, 1e-5), (-1e5, 1e5)]  # Adjust the limits according to your data

        # Use differential_evolution to minimize the objective function
        result = differential_evolution(
            objective,
            bounds,
            seed=42,               # Set the seed for consistent results
            maxiter=4000,          # Increase the number of iterations
            popsize=20,            # Increase population size
            mutation=(0.5, 1),     # Adjust the mutation factor
        )

        # Obtain the optimal values ​​of 'm' and 'b'
        m_opt, b_opt = result.x

        # Calculate the values ​​of the adjusted function for the entire row
        fitted_line = m_opt * x_values + b_opt

        # Subtract the function adjusted to the current line
        Flatten_img[i, :] = y_values - fitted_line

        # # Adjust the median from the line to zero
        # median_selected = np.median (flatten_img [i,:])
        # Flatten_img [i,:] -= median_selected

    return Flatten_img, mask_selected

def FlattenPlus_filtered(img, lower_perc=10, upper_perc=10):
    def linear_model(x, a):
        return a * x

    Flatten_img = img.copy()
    mask_selected = np.zeros_like(img, dtype=bool)  # Initialize the selection mask

    # Apply the adjustment to the lines
    for i in range(0, Flatten_img.shape[0]):
        # Select the points of the current line
        line_current = Flatten_img[i, :]

        # Order the points of the current line by value
        sorted_indices = np.argsort(line_current)
        n_points = len(line_current)

        # Discard the lower and upper percentage
        lower_bound = int(n_points * lower_perc / 100)
        upper_bound = int(n_points * (100 - upper_perc) / 100)

        # Select the intermediate points
        selected_indices = sorted_indices[lower_bound:upper_bound]
        x_values = np.arange(n_points)[selected_indices]
        y_values = line_current[selected_indices]

        # Mark the selected points in the mask
        mask_selected[i, selected_indices] = True

        # Define the objective function that minimizes the standard deviation
        def objective(params):
            a = params[0]
            fitted_line = linear_model(x_values, a)
            return np.std(y_values - fitted_line)

        # Define the limits for parameter 'A'
        bounds = [(-1e-5, 1e-5)]  # Parameter limits

        # Use differential_evolution to minimize the objective function
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42,                 # Set the seed for consistent results
            maxiter=4000,            # Increase the number of iterations
            popsize=20,              # Increase population size
            mutation=(0.5, 1),       # Adjust the mutation factor
        )

        # Get the optimal value of 'a'
        a_opt = result.x[0]

        # Calculate the values ​​of the adjusted function for the entire row
        fitted_line = linear_model(np.arange(n_points), a_opt)

        # Subtract the function adjusted to the current line
        Flatten_img[i, :] = line_current - fitted_line

        # Adjust the median of the selected points to zero
        median_selected = np.median(Flatten_img[i, selected_indices])
        Flatten_img[i, :] -= median_selected

    return Flatten_img, mask_selected

def FlattenPlus_filtered_v3(img, lower_perc=10, upper_perc=10):
    # Define the objective function outside the loop
    def objective(params, x_values, y_values):
        m, b = params
        fitted_line = m * x_values + b
        squared_errors = (y_values - fitted_line) ** 2
        std_squared_errors = np.std(squared_errors)
        return std_squared_errors
    Flatten_img = img.copy()
    mask_selected = np.zeros_like(img, dtype=bool)  # Initialize the selection mask

    # Apply the adjustment to each row of the image
    for i in range(Flatten_img.shape[0]):
        # Select the points of the current line
        line_current = Flatten_img[i, :]
        n_points = len(line_current)

        # Order the points of the current line
        sorted_indices = np.argsort(line_current)
        
        # Calculate lower and upper limits to rule out the percentage
        lower_bound = int(n_points * lower_perc / 100)
        upper_bound = int(n_points * (100 - upper_perc) / 100)
        
        # Select the intermediate points according to the specified percentiles
        selected_indices = sorted_indices[lower_bound:upper_bound]
        x_values = np.arange(n_points)[selected_indices]
        y_values = line_current[selected_indices]

        # Mark the selected points in the mask
        mask_selected[i, selected_indices] = True

        # Define the limits for the 'm' and 'b' parameters
        bounds = [(-1e-5, 1e-5), (-1e5, 1e5)]  # Adjust the limits according to your data

        # Use differential_evolution to minimize the objective function
        result = differential_evolution(
            objective, 
            bounds, 
            args=(x_values, y_values),  # We pass the specific values ​​as arguments
            seed=42,               # Set the seed for consistent results
            maxiter=4000,          # Maximum number of iterations
            popsize=20,            # Population size
            mutation=(0.5, 1),     # Adjust the mutation factor
        )

        # Obtain the optimal values ​​of 'm' and 'b'
        m_opt, b_opt = result.x

        # Calculate the values ​​of the adjusted function for the entire row
        fitted_line = m_opt * np.arange(n_points) + b_opt

        # Subtract the function adjusted to the current line
        Flatten_img[i, :] = line_current - fitted_line

        # Adjust the median line to zero
        median_selected = np.median(Flatten_img[i, selected_indices])
        Flatten_img[i, :] -= median_selected
    
    def adjust_lowest_median_to_zero(image, percentage_lowest):
        
        # Subtract the adjusted plane from the image
        image_corrected = image

        # Flatten the corrected image to easily operate with values
        image_flat = image_corrected.flatten()

        # Order the corrected image values ​​and select the lowest percentage
        sorted_indices = np.argsort(image_flat)
        n_points = int(len(image_flat) * percentage_lowest)
        lowest_indices = sorted_indices[:n_points]

        # Obtain the values ​​corresponding to the lowest indices
        lowest_values = image_flat[lowest_indices]
    
        # Calculate the average of the lowest values
        median_lowest = np.median(lowest_values)
        
        # Adjust the image so that the lowest values ​​have zero medium
        image_corrected -= median_lowest

        return image_corrected

    Flatten_img = adjust_lowest_median_to_zero(Flatten_img, 0.1)
    return Flatten_img, mask_selected

def equalize_median_with_std_derivative_threshold(img, factor):
    """
    Applies an offset to each line of the image so that the median of the line
    current is equal to the median of the previous line, discarding the points
    whose derivative in absolute value is greater than a threshold based on the 
    standard deviation of the derivative of the line.
    
    Args:
        img (numpy.ndarray): Image to which the row offset is applied.
        factor (float): Multiplicative factor of the standard deviation of the 
                        derived. The threshold will be calculated as a factor * std(derivative).
    
    Returns:
        numpy.ndarray: Adjusted image with offsets applied.
    """
    
    adjusted_img = img.copy()

    # Process line per line
    for i in range(1, adjusted_img.shape[0]):  # We start from the second row
        # Obtain the current line and the previous
        current_line = adjusted_img[i, :]
        previous_line = adjusted_img[i - 1, :]

        # Calculate the derived from the current line
        derivative = np.abs(np.diff(current_line))

        # Calculate the threshold as a multiplicative factor of the standard deviation of the derivative
        std_derivative = np.std(derivative)
        threshold = factor * std_derivative

        # Create a mask that discards the points with a derivative greater than the threshold
        valid_mask = np.ones_like(current_line, dtype=bool)
        valid_mask[1:][derivative > threshold] = False  # Discard points with high derivative

        # Calculate the median of the valid points (without discarded) for the current line and the previous
        current_median = np.median(current_line[valid_mask])
        previous_median = np.median(previous_line)

        # Calculate the offset to match the median of the current line with the previous
        offset = previous_median - current_median

        # Apply offset to the current line
        adjusted_img[i, :] += offset

    return adjusted_img


# ********************** AUTOMATIC FUNCTIONS ************************

# Function that returns the image flatten by diff + optimal planet and plus
def AutoFlattenPlus(img,perc = 0.4):
    imgDiff = FitOffsetToFlattingImageByDiff(img)
    image_corrected, fitted_plane = SubtractOptimalPlane(imgDiff,0.1,show=False)
    img_plus, mask=FlattenPlus(image_corrected,perc)
    return img_plus, mask

# Auto Flatten function for samples with little substrate or containing areas where there are lines without substrate or very little substrate
def AutoFlattenByOptimalPlane(img, perc = 0.1):
    imgDiff = FitOffsetToFlattingImageByDiff(img)
    image_corrected, fitted_plane = SubtractOptimalPlane(imgDiff,perc,show=False)
    return image_corrected
     
# Function to show images
def display_images(img1, img2, img_combined,titles):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show the first image with color scale
    im1 = axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0])  # Add the color bar to the first image

    # Show the second image with color scale
    im2 = axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1])  # Add the color bar to the second image

    # Show the combined image with color scale
    im3 = axes[2].imshow(img_combined, cmap='gray')
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2])  # Add the color bar to the third image

    plt.show()


if __name__ == '__main__':
    pth= r'C:\Users\z78\Documents\Captures\DriveAFM\2024\Oct\21th'
    fn = 'Adama_Pa_2_16.gwy'
    img = mgf.LoadChannelGWY_FileToImage(filename = fn, path=pth)
    image_corrected, mask = SubtracGlobalPlaneManualByPoints(img.matriz, show=True)
    SaveMatriz(image_corrected, 'TestImage_der.txt', '', imgPro = None)