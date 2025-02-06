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

import os
import cv2
import re
import numpy as np
from skimage.transform import resize
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import gwyfile
import flattening_v2 as fla
import managefiles as mgf
from stitch2d import create_mosaic, StructuredMosaic
from AFMclasses import clImage, ChannelType, ExtentionType

def combinar_imagenes_grid(imagenes, filas, columnas, solapamiento_porcentaje, withOffset=False):
    """
    Combines a list of clImage objects in a grid format with specified overlap.
    """
    alto, ancho = imagenes[0].matriz.shape
    for img in imagenes:
        if img.matriz.shape != (alto, ancho):
            raise ValueError("All images must have the same size.")
    
    ancho_final = int(ancho * (columnas - solapamiento_porcentaje * (columnas - 1)))
    alto_final = int(alto * (filas - solapamiento_porcentaje * (filas - 1)))
    size_x_original = imagenes[0].size_x
    size_y_original = imagenes[0].size_y
    size_x_final = size_x_original * (ancho_final / ancho)
    size_y_final = size_y_original * (alto_final / alto)
    imagen_combinada = np.zeros((alto_final, ancho_final), dtype=np.float32)
       
    for i in range(filas):
        for j in range(columnas):
            idx = (filas - 1 - i) * columnas + j
            img = imagenes[idx].matriz
            x_offset = int(j * ancho * (1 - solapamiento_porcentaje))
            y_offset = int(i * alto * (1 - solapamiento_porcentaje))
            y_end, x_end = y_offset + alto, x_offset + ancho
            if withOffset:
                offsets = []
                if i > 0:
                    area_superpuesta = imagen_combinada[y_offset:y_end, x_offset:x_offset + int(ancho * solapamiento_porcentaje)]
                    offsets.append(np.mean(area_superpuesta - img[:, :int(ancho * solapamiento_porcentaje)]))
                if j > 0:
                    area_superpuesta = imagen_combinada[y_end - int(alto * solapamiento_porcentaje):y_end, x_offset:x_end]
                    offsets.append(np.mean(area_superpuesta - img[-int(alto * solapamiento_porcentaje):, :]))
                offset = np.mean(offsets) if offsets else 0
                img += offset
            imagen_combinada[y_offset:y_end, x_offset:x_end] = img
    
    nuevo_climage = clImage(
        channel=imagenes[0].channel,
        size_x=size_x_final,
        size_y=size_y_final,
        unitxy=imagenes[0].unitxy,
        unitz=imagenes[0].unitz,
        offset_x=0.0,
        offset_y=0.0,
        lenpxx=ancho_final,
        lenpxy=alto_final,
        matriz=imagen_combinada
    )
    
    return nuevo_climage

def ResizePerc(matrix, scale_percent=20):
    """
    Resizes a matrix by a given scale percentage.
    """
    original_height, original_width = matrix.shape[:2]
    new_shape = (int(original_height * scale_percent / 100), int(original_width * scale_percent / 100))
    return resize(matrix, new_shape)

def SimpleCoordinateMethod(PathXYZFiles=None, filas=None, columnas=None, solapamiento_porcentaje=0.1, nstd=2, withOffset=False):
    """
    Creates a mosaic using images from a directory, arranged in a grid with specified overlap.
    """
    if PathXYZFiles is None:
        PathXYZFiles = mgf.OpenFolderDialog()
    Image_list = mgf.LoadAllImageFile_fromDirectory(Directory=PathXYZFiles, Exttype='.xyz')
    out_folder = os.path.join(PathXYZFiles, 'out_stitching')
    os.makedirs(out_folder, exist_ok=True)
    if filas is None or columnas is None:
        nimg = len(Image_list)
        filas = columnas = int(np.sqrt(nimg))
    combinada = combinar_imagenes_grid(Image_list, filas, columnas, solapamiento_porcentaje, withOffset=withOffset)
    mgf.SaveImageToXYZ(combinada, filename='SimpleCoordinate_mosaic.xyz', path=out_folder)
    mgf.SaveNumpyToPNG_By_PIL(combinada.matriz, out_folder, 'SimpleCoordinate_mosaic.png', norm=None, nstd=nstd, colormap='copper')

def ecualizar_imagenes_en_directorio(directorio_origen, directorio_destino):
    """
    Equalizes the histogram of all images in a directory and saves the results to a destination directory.
    """
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)
    for archivo in os.listdir(directorio_origen):
        ruta_completa_origen = os.path.join(directorio_origen, archivo)
        if os.path.isfile(ruta_completa_origen):
            imagen = cv2.imread(ruta_completa_origen, cv2.IMREAD_GRAYSCALE)
            if imagen is not None:
                imagen_ecualizada = cv2.equalizeHist(imagen)
                ruta_completa_destino = os.path.join(directorio_destino, archivo)
                cv2.imwrite(ruta_completa_destino, imagen_ecualizada)

def load_images_from_folder(folder_path):
    """
    Loads PNG files from a folder into a list sorted by the numeric index in the file name.
    """
    images = []
    pattern = re.compile(r'_(\d+)\.png$')
    files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith('.png') and pattern.search(f)],
        key=lambda x: int(pattern.search(x).group(1))
    )
    for file in files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('L')
        images.append(np.array(img))
    return images

def display_image(image, title="Combined Image"):
    """
    Displays a given image with an optional title.
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    directory = r'C:\Users\z78\Documents\Projects\AppsPy\Stitching\Out_flagella_2024_11_05_16_00_36'
    SimpleCoordinateMethod(PathXYZFiles=directory, filas=3, columnas=16, nstd=2)
    print('End')
