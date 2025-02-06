"""
Author: Ruben Millan-Solsona
Date of Creation: August 2024

Description:
This module provides utility functions for handling AFM image data, including file dialogs,
directory management, image parameter saving/loading, image normalization, and visualization.

Dependencies:
- os
- numpy
- typing
- datetime
- AFMclasses (contains clImage, ChannelType, ExtentionType)
- gwyfile (for loading Gwyddion data files)
- matplotlib.pyplot (for image display)
- matplotlib.colors (for color normalization)
- scipy.interpolate (for custom color map interpolation)
- PIL.Image (for image manipulation)
- tkinter (for file dialogs)

"""


import os
import numpy as np
from typing import List, Tuple
from datetime import datetime
from AFMclasses import clImage, ChannelType, ExtentionType
import gwyfile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from scipy.interpolate import interp1d
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Opens a file dialog to select a file, defaults to the project directory
def OpenFileDialog(default_path=None, ext_file=[("All files", "*.*")]):
    if default_path is None:
        default_path = os.getcwd()
    root = tk.Tk()
    root.withdraw()
    selected_file = filedialog.askopenfilename(
        title="Open file",
        filetypes=ext_file,
        defaultextension=ext_file[0][1],
        initialdir=default_path
    )
    return selected_file if selected_file else None

# Opens a file dialog to save a file, defaults to the project directory
def SaveFileDialog(default_path=None, ext_file=[("All files", "*.*")]):
    if default_path is None:
        default_path = os.getcwd()
    root = tk.Tk()
    root.withdraw()
    file_to_save = filedialog.asksaveasfilename(
        title="Save File as",
        defaultextension=ext_file[0][1],
        filetypes=ext_file,
        initialdir=default_path
    )
    return file_to_save if file_to_save else None

# Opens a folder dialog to select a directory, defaults to the project directory
def OpenFolderDialog(default_path=None):
    if default_path is None:
        default_path = os.getcwd()
    root = tk.Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory(
        title="Select a directory",
        initialdir=default_path
    )
    return os.path.normpath(selected_dir) if selected_dir else None

# Creates a directory with a timestamp in its name, default to current project folder
def DoFolderWithDate(label="", root_folder=None):
    current_time = datetime.now()
    folder_name = current_time.strftime(f"Out_{label}_%Y_%m_%d_%H_%M_%S")
    if root_folder is None:
        root_folder = os.getcwd()
    folder_path = os.path.join(root_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return os.path.abspath(folder_path)

# Saves image parameters to a .txt file without saving the data matrix
def SaveParmaImageTotxt(image_obj, filename="imageparameters.txt", path=None):
    if path is None:
        path = os.getcwd()
    filename = f"{os.path.splitext(filename)[0]}.txt"
    dir_path = os.path.join(path, filename)
    with open(dir_path, 'w') as file:
        file.write(f"Channel: {image_obj.channel}\n")
        file.write(f"Size X: {image_obj.size_x:.8e}\n")
        file.write(f"Size Y: {image_obj.size_y:.8e}\n")
        file.write(f"Offset X: {image_obj.offset_x:.8e}\n")
        file.write(f"Offset Y: {image_obj.offset_y:.8e}\n")
        file.write(f"Unit XY: {image_obj.unitxy}\n")
        file.write(f"Unit Z: {image_obj.unitz}\n")
        file.write(f"len X: {image_obj.lenpxx}\n")
        file.write(f"len Y: {image_obj.lenpxy}\n")
        file.write(f"Tag: {image_obj.tag}\n")

# Reads parameters from a .txt file to create a clImage object with a zeroed matrix
def ReadParametersImageTXT(filename, path=None) -> clImage:
    if path is None:
        path = os.getcwd()
    filename += ".txt" if not filename.endswith(".txt") else ""
    file_path = os.path.join(path, filename)
    params = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if ":" in line:
                key, value = line.split(":")
                params[key.strip()] = value.strip()
    lenpxx = int(params.get("lenpxx", 256))
    lenpxy = int(params.get("lenpxy", 256))
    matrix = np.zeros((lenpxx, lenpxy))
    return clImage(
        channel=params.get("channel", "Backward - Topography"),
        path=path,
        filename=filename,
        size_x=float(params.get("Size X", 1.0)),
        size_y=float(params.get("Size Y", 1.0)),
        unitxy=params.get("Unit XY", "µm"),
        unitz=params.get("Unit Z", "nm"),
        offset_x=float(params.get("Offset X", 0.0)),
        offset_y=float(params.get("Offset Y", 0.0)),
        lenpxx=lenpxx,
        lenpxy=lenpxy,
        tag=params.get("Tag", ""),
        matriz=matrix
    )

# Saves a matrix to an XYZ file format with scientific notation
def SaveImageToXYZ(image_obj, filename=None, path=None):
    if filename is None:
        filename = f"{image_obj.channel}.xyz"
    else:
        filename = f"{os.path.splitext(filename)[0]}.xyz"
    if path is None:
        path = os.getcwd()
    dir_path = os.path.join(path, filename)
    step_x = image_obj.size_x / image_obj.matriz.shape[0]
    step_y = image_obj.size_y / image_obj.matriz.shape[1]
    with open(dir_path, 'w') as file:
        file.write("WSxM file copyright RMS\nWSxM ASCII XYZ file\n")
        for i in range(image_obj.matriz.shape[0]):
            for j in range(image_obj.matriz.shape[1]):
                x_real = i * step_x
                y_real = j * step_y
                file.write(f"{x_real:.8e} {y_real:.8e} {image_obj.matriz[i, j]:.8e}\n")

# Shows a clImage object's image with a color scale and title
def DisplayImage_with_scale(image_obj: clImage, title=None, color_map='copper', std_devs=4):
    image_data = image_obj.matriz
    mean_val = np.mean(image_data)
    std_val = np.std(image_data)
    vmin = mean_val - std_devs * std_val
    vmax = mean_val + std_devs * std_val
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(image_data, cmap=color_map, vmin=vmin, vmax=vmax,
                   extent=[0, image_obj.size_x, 0, image_obj.size_y])
    ax.set_title(f"{title}" if title else f"{image_obj.channel} - {image_obj.tag}")
    ax.set_xlabel(f"X [{image_obj.unitxy}]")
    ax.set_ylabel(f"Y [{image_obj.unitxy}]")
    fig.colorbar(im, ax=ax, label=f"Z [{image_obj.unitz}]")
    plt.show(block=False)

if __name__ == '__main__':
    images_list = LoadAllImageFile_fromDirectory(Directory=None, Exttype='.gwy')
    DisplayImage_with_scale(images_list)
    plt.pause(0)
