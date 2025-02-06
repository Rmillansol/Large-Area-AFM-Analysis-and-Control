"""
Author: Ruben Millan-Solsona
Date of Creation: August 2024

Description:
This module provides various functions to stitch and process AFM images,
including grid combination, resizing, coordinate-based and structured mosaic methods,
histogram equalization, image derivation, and thresholding for masks.

Dependencies:
- os
- numpy
- re
- cv2
- gwyfile
- matplotlib.pyplot
- matplotlib.colors
- PIL.Image
- skimage.transform.resize
- scipy.ndimage.shift
- stitch2d (for creating and aligning mosaics)
- AFMclasses (contains clImage, ChannelType, ExtentionType)
- managefiles and flattening_v2 (custom modules for file management and image processing)
"""
# From Stitch2D IMPORT Stitcher2D
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import Normalize
# IMPORT MATPLOTLIB.COLORS AS MCOLORS
import gwyfile
import flattening_v2 as fla
import managefiles as mgf
from PIL import Image
import managefiles as mf
from stitch2d import create_mosaic
from stitch2d import StructuredMosaic
from AFMclasses import clImage, ChannelType, ExtentionType
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
import re


def combinar_imagenes_grid(imagenes,  filas, columnas, solapamiento_porcentaje,withOffset = False):
    # All images are climage objects, so we access the array from img.array
    alto, ancho = imagenes[0].matriz.shape
    
    # Verify that all images are The Same Size
    for img in imagenes:
        if img.matriz.shape != (alto, ancho):
            raise ValueError("All images must be the same size.")
    
    # Calculate The Size of the Final Image In Pixels
    ancho_final = int(ancho * (columnas - solapamiento_porcentaje * (columnas - 1)))
    alto_final = int(alto * (filas - solapamiento_porcentaje * (filas - 1)))
    
    # Calculate The Size of the Final Image in the Original Units (E.G. µm)
    # We assume that the pixel size is the Same in All images
    size_x_original = imagenes[0].size_x
    size_y_original = imagenes[0].size_y
    size_x_final = size_x_original * (ancho_final / ancho)
    size_y_final = size_y_original * (alto_final / alto)
    
    # Create Blank image for the combined image and its mask
    imagen_combinada = np.zeros((alto_final, ancho_final), dtype=np.float32)
       
    for i in range(filas):
        for j in range(columnas):
            # The Image Index for Bottom to Top and Left to Right Order
            idx = (filas - 1 - i) * columnas + j
            img = imagenes[idx].matriz
                       
            # Calculate Starting Position to Paste Image
            x_offset = int(j * ancho * (1 - solapamiento_porcentaje))
            y_offset = int(i * alto * (1 - solapamiento_porcentaje))
            
            # Calculate The Area Where The Image Will Be Pasted
            y_end = y_offset + alto
            x_end = x_offset + ancho
            if y_end > alto_final or x_end > ancho_final:
                raise ValueError("The calculation of the overlap or the final size is incorrect.")
            if withOffset:
                offsets = []
                if i > 0:  # Left
                    area_superpuesta = imagen_combinada[y_offset:y_end, x_offset:x_offset + int(ancho * solapamiento_porcentaje)]
                    offsets.append(np.mean(area_superpuesta - img[:, :int(ancho * solapamiento_porcentaje)]))
                else:
                    offsets.append(0)

                if j > 0:  # Below
                    area_superpuesta = imagen_combinada[y_end - int(alto * solapamiento_porcentaje):y_end, x_offset:x_end]
                    offsets.append(np.mean(area_superpuesta - img[-int(alto * solapamiento_porcentaje):, :]))
                else:
                    offsets.append(0)
                                            
                if offsets:
                    # Calculate the offset as the average of the differences in the overlapping regions
                    offset = np.mean(offsets)
                    img += offset
            
            # Paste the image and mask at the calculated position
            imagen_combinada[y_offset:y_end, x_offset:x_end] = img
           
    
    # image_chombine, plane_fitted = fla.subtractplanebymask (image_chombin, mask_chombine, show = 0)
    
    # Create The New Climage Object for The Combined Image
    nuevo_climage = clImage(
        channel=imagenes[0].channel,
        size_x=size_x_final,
        size_y=size_y_final,
        unitxy=imagenes[0].unitxy,  # Keep the Original Units
        unitz=imagenes[0].unitz,
        offset_x=0.0,
        offset_y=0.0,
        lenpxx=ancho_final,  # New Width in Pixels
        lenpxy=alto_final,   # New Pixel Height
        matriz=imagen_combinada
    )
    
    return nuevo_climage

def ResizePerc(Matriz,scale_percent =20):
    
    original_height, original_width = Matriz.shape[:2]  # Get Original Dimensions

    # Calculate New Size
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)
    new_shape = (new_height, new_width)

    # Resize The Image
    resized_Matriz = resize(Matriz, new_shape)
    return resized_Matriz

def SimpleCoordinateMethod(PathXYZFiles = None,filas = None,columnas = None,solapamiento_porcentaje = 0.1, nstd = 2, withOffset = False):
    if PathXYZFiles is None:
        PathXYZFiles= mgf.OpenFolderDialog()
    # Function That performs The Mosaic from A Path That Contains All The Png Images
    Image_list = mgf.LoadAllImageFile_fromDirectory(Directory = PathXYZFiles, Exttype = '.xyz')
    # Define The Output Path for The Tile
    out_folder = os.path.join(PathXYZFiles, 'out_stitching_Coordinate_Method')

    # Create 'out_stitching' folder if it does not exist
    os.makedirs(out_folder, exist_ok=True)

    # I check if it is square i calculate it
    if filas is None or columnas is None:
        nimg= len(Image_list)
        filas = int(np.sqrt(nimg))
        columnas = filas

    combinada= combinar_imagenes_grid(Image_list, filas, columnas, solapamiento_porcentaje, withOffset=False)
    # Combined = adjust_y_combinar_imagenes_grid (image_list, rows, columns, overlapping_porception, adjustment_porcess = 0.05)
    mgf.SaveImageToXYZ(combinada, filename = 'Coordinate_mosaic.xyz', path =  out_folder)
    mgf.SaveNumpyToPNG_By_PIL(combinada.matriz, out_folder, 'Coordinate_mosaic.png', norm = None, nstd = nstd, colormap='copper')

    return  combinada

def SimpleStitch2dMethod(PathPNGFiles = None,dimX = 3, downsample = 0.9,limit = None, from_placed = True):
# Function That performs The Mosaic from A Path That Contains All The Png Images
    if PathPNGFiles is None:
        PathPNGFiles= mgf.OpenFolderDialog()

    # #Lee image parameters
    # IMGE0 = mgf.readparametersimagetxt (fillename = 'imageparameters.txt', path = pathpngfiles)
    # Imge0.info_class ()
     # Define the output route for the mosaic
    out_folder = os.path.join(PathPNGFiles, 'out_stitching_2D')
    out_path = os.path.join(out_folder, 'SinmpleStitch2d_mosaic.jpg')
    out_param_path = os.path.join(out_folder, 'params.json')
    # Create 'out_stitching' folder if it does not exist
    os.makedirs(out_folder, exist_ok=True)

    mosaic = StructuredMosaic( 
    PathPNGFiles,
    dim=dimX,                 # Number of Tiles in Primary Axis
    origin="lower left",       # Position of First Tile
    direction="horizontal",    # Primary axis (I.E., The Direction to Traverse First)
    pattern="raster"           # Snake or Raster
  )
    mosaic.downsample(downsample)
    if limit is None:
         mosaic.align()
    else:
        mosaic.align(limit=limit)

    # Build The Rest of the Mosaic Based on the postitioned tiles.If from_placed
    # Is True, Missing Tiles are appended to the already posited tiles.If
    # False, a new mosaic is calculated from scratch.
    mosaic.build_out(from_placed=from_placed)

    mosaic.reset_tiles()
    print(out_folder)
    # mosaic.save_params (out_param_path)
    # mosaic.load_params (out_param_path)

    img:np = mosaic.stitch()   # The numpy matrix returns
    mosaic.save_params(out_param_path)
    # ranks, columns = img.shape [: 2]

    # print (rows, columns)
    # mosaic.smooth_seams ()
    mosaic.save(out_path)
    
    return img

def ecualizar_imagenes_en_directorio(directorio_origen, directorio_destino):
    # CREATE THE DESTINATION FOLDER IF IT DOES NOT EXIST
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    # Iterate over all files in source directory
    for archivo in os.listdir(directorio_origen):
        ruta_completa_origen = os.path.join(directorio_origen, archivo)

        # ONLY PROCESS IF IT IS AN IMAGE File
        if os.path.isfile(ruta_completa_origen):
            # Read the image
            imagen = cv2.imread(ruta_completa_origen, cv2.IMREAD_GRAYSCALE)  # EQUALIZATION TYPICALLY ON GRAYSCALE IMAGES
            if imagen is None:
                print(f"Could not read the image: {archivo}")
                continue

            # Apply histogram equalization
            imagen_ecualizada = cv2.equalizeHist(imagen)

            # Save the Equalized Image in the Destination Folder with the Same Name
            ruta_completa_destino = os.path.join(directorio_destino, archivo)
            cv2.imwrite(ruta_completa_destino, imagen_ecualizada)

    print(f"Equalized images saved in {directorio_destino}")

def derivada_y_normalizar_imagenes(directorio_origen, directorio_destino):
    # CREATE THE DESTINATION FOLDER IF IT DOES NOT EXIST
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    # Iterate over all files in source directory
    for archivo in os.listdir(directorio_origen):
        ruta_completa_origen = os.path.join(directorio_origen, archivo)

        # ONLY PROCESS IF IT IS AN IMAGE File
        if os.path.isfile(ruta_completa_origen):
            # Read the image
            imagen = cv2.imread(ruta_completa_origen, cv2.IMREAD_GRAYSCALE)  # Work with grays
            if imagen is None:
                print(f"No se pudo leer la imagen: {archivo}")
                continue

            # Calculate the derivative by rows (derived in the X axis)
            derivada = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)

            # Normalize the derivative so that the values ​​are between 0 and 255
            derivada_normalizada = np.zeros_like(derivada)
            cv2.normalize(derivada, derivada_normalizada, 0, 255, cv2.NORM_MINMAX)

            # Convert Uint8 data type (gray scale image)
            derivada_normalizada = np.uint8(derivada_normalizada)

            # Save the normalized derived image in the destination folder with the same name
            ruta_completa_destino = os.path.join(directorio_destino, archivo)
            cv2.imwrite(ruta_completa_destino, derivada_normalizada)

    print(f"Imágenes derivadas y normalizadas guardadas en {directorio_destino}")

def derivada_normalizada_y_guardar_mascara(directorio_origen, directorio_destino, threshold):
    # Create the destination folder if there is no
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    # Iterate on all files in the directory of origin
    for archivo in os.listdir(directorio_origen):
        ruta_completa_origen = os.path.join(directorio_origen, archivo)

        # Just process if it is an image file
        if os.path.isfile(ruta_completa_origen):
            # Read the image
            imagen = cv2.imread(ruta_completa_origen, cv2.IMREAD_GRAYSCALE)  # Work with grays
            if imagen is None:
                print(f"No se pudo leer la imagen: {archivo}")
                continue

            # Calculate the derivative by rows (derived in the X axis)
            derivada = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)

            # Normalize the derivative so that the values ​​are between 0 and 255
            derivada_normalizada = np.zeros_like(derivada)
            cv2.normalize(np.abs(derivada), derivada_normalizada, 0, 255, cv2.NORM_MINMAX)

            # Convert Uint8 data type (gray scale image)
            derivada_normalizada = np.uint8(derivada_normalizada)

            # Apply a threshold (threshold) to the standardized derivative to create a binary mask
            _, mascara = cv2.threshold(derivada_normalizada, threshold, 255, cv2.THRESH_BINARY)

            # Save the mask in the fate folder with the same name
            ruta_completa_destino = os.path.join(directorio_destino, archivo)
            cv2.imwrite(ruta_completa_destino, mascara)

    print(f"Máscaras derivadas y umbralizadas guardadas en {directorio_destino}")

def load_images_from_folder(folder_path):
    """
    Reads PNG files from a folder and places them in a list in ascending order,
    based on the number of the file name in FileName_# .png format.
    
    Parameters:
    - folder_path: Path of the folder that contains the images.
    
    Returns:
    - images: List of images in NumPy format, ordered by numerical index.
    """
    images = []
    
    # REGEX to find the number in names with format Namearchivo _#. PNG
    pattern = re.compile(r'_(\d+)\.png$')
    
    # Filter and order PNG files depending on the number extracted from the name
    files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith('.png') and pattern.search(f)],
        key=lambda x: int(pattern.search(x).group(1))  # Order for the number
    )
    
    # Read each file and add it to the image list
    for file in files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('L')  # Convert to gray scale if necessary
        images.append(np.array(img))  # Convert the image to NUMPY format and add to the list
    
    return images

def find_optimal_position(img1, img2, initial_overlap, max_shift=5):
    """
   Find the optimal position of img2 with respect to img1 to minimize the error in the overlap area.
    """
    best_shift = (0, 0)
    min_error = float('inf')
    
    # Determine the overlap region in X
    overlap_x = int(initial_overlap * img1.shape[1])
    
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            # Shift IMG2 and calculate the difference in the overlap region
            shifted_img2 = shift(img2, shift=(dy, dx))
            
            # Extract the overlap areas of both images
            region1 = img1[:, -overlap_x:]
            region2 = shifted_img2[:, :overlap_x]
            
            # Calculate the average quadratic error in the overlap area
            error = np.mean((region1 - region2) ** 2)
            
            if error < min_error:
                min_error = error
                best_shift = (dx, dy)
    
    return best_shift

def stitch_images_grid(images, grid_shape, overlap=0.1):
    """
    Combine a list of images in a grid with position adjustment in rows and columns,
    considering the order from left to right and from bottom to top.
    """
    rows, cols = grid_shape
    img_height, img_width = images[0].shape
    overlap_x = int(overlap * img_width)
    overlap_y = int(overlap * img_height)

    # Create an empty dog ​​for the Grid of images
    final_image = np.zeros((rows * img_height, cols * img_width), dtype=np.float32)

    # Place the first image in the lower left corner of the grid
    final_image[(rows - 1) * img_height : rows * img_height, 0:img_width] = images[0]

    # Fill the Grid from the bottom up and from left to right
    for row in range(rows):
        for col in range(cols):
            if row == 0 and col == 0:
                continue  # Jump the first image already placed

            # Calculate the current image index considering the order
            img_idx = (rows - 1 - row) * cols + col
            current_image = images[img_idx]

            # Place the image according to the previous image in the row or column
            if col > 0:  # Align horizontally with the previous image in the row
                left_image = final_image[(rows - 1 - row) * img_height : (rows - row) * img_height, 
                                         (col - 1) * img_width : col * img_width]
                dx, dy = find_optimal_position(left_image, current_image, overlap)
                adjusted_image = shift(current_image, shift=(dy, dx))
                final_image[(rows - 1 - row) * img_height : (rows - row) * img_height, 
                            col * img_width : (col + 1) * img_width] = adjusted_image

            elif row < rows - 1:  # Align vertically with the previous image in the column
                top_image = final_image[(rows - row) * img_height : (rows - row + 1) * img_height, 
                                        col * img_width : (col + 1) * img_width]
                dx, dy = find_optimal_position(top_image, current_image, overlap)
                adjusted_image = shift(current_image, shift=(dy, dx))
                final_image[(rows - 1 - row) * img_height : (rows - row) * img_height, 
                            col * img_width : (col + 1) * img_width] = adjusted_image

    return final_image

def display_image(image, title="Imagen Combinada"):
    """
   Displays the resulting image in a window with a title.
    
    Parameters:
    - image: The image in NumPy array format.
    - title: Title of the image window (optional).
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")  # Hides the axes
    plt.show()


if __name__ == '__main__':
    directorio = r'C:\Users\z78\Documents\Projects\AppsPy\Stitching\Out_flagella_2024_11_05_16_00_36'

    SimpleStitch2dMethod(PathPNGFiles = directorio,dimX = 4, downsample = 1,limit = None, from_placed = False)
    # Simplocoordinatemethod (Pathxyzfiles = directory, rows = 3, columns = 16, nstd = 2)
    print('end')
    