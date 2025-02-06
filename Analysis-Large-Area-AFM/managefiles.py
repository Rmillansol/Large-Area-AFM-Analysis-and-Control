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

def OpenFileDialog(default_path=None,ext_file: list = [("All files", "*.*")]):
   
    # Use the project route if a default_path is not provided
    if default_path is None:
        default_path = os.getcwd()

    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()

    # Open the dialog box to select a file
    archivo_seleccionado = filedialog.askopenfilename(
        title="Open file",
        filetypes=ext_file,
        defaultextension=ext_file[0][1],  # Predetermined extension
        initialdir=default_path  # Default route
    )
    
    if archivo_seleccionado:  # Verify if a file was selected
        return archivo_seleccionado
    else:
        return None  # Return None if you are canceled or not selected file

def SaveFileDialog(default_path=None,ext_file: list = [("All files", "*.*")]):
    # Use the project route if a default_path is not provided
    if default_path is None:
        default_path = os.getcwd()

    # Create a root window
    root = tk.Tk()
    root.withdraw()

    # Open the dialog box to save a file
    archivo_a_guardar = filedialog.asksaveasfilename(
        title="Save File as",
        defaultextension=ext_file[0][1],  # Predetermined extension
        filetypes=ext_file,
        initialdir=default_path  # Default route
    )
    
    if archivo_a_guardar:  # Verify if a file was selected to save
        return archivo_a_guardar
    else:
        return None  # Return None if canceled

def OpenFolderDialog(default_path=None):
    # Use the project route if a default_path is not provided
    if default_path is None:
        default_path = os.getcwd()

    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()

    # Open the dialog box to select a directory
    directorio_seleccionado = filedialog.askdirectory(
        title="Seleccionar un directorio",
        initialdir=default_path  # Default route
    )
    # Normalize the route to be compatible with the operating system
    directorio_normalizado = os.path.normpath(directorio_seleccionado)
    if directorio_normalizado:  # Verify if a directory was selected
        return directorio_normalizado
    else:
        return None  # Return None if canceled

def DoFolderWithDate(etiqueta="", carpeta_raiz=None):
    # Get the current date and time
    current_time = datetime.now()

    # Format the date and time as year_mes_dia_horas_min_seconds
    folder_name = current_time.strftime(f"Out_{etiqueta}_%Y_%m_%d_%H_%M_%S")

    # If a root folder is not provided, use the current project folder
    if carpeta_raiz is None:
        carpeta_raiz = os.getcwd()

    # Create the complete path of the folder
    folder_path = os.path.join(carpeta_raiz, folder_name)

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    # Return the absolute path of the folder
    return os.path.abspath(folder_path)

def SaveParmaImageTotxt(image_obj, filename: str = None, path: str = None):
    # function that saves the parameters of an image in a TXT file without saving the data matrix
    # If you are None, we use the channel as the file name
    if filename is None:
        filename = "imageparameters.txt"
    else:
        filename_without_ext = os.path.splitext(filename)[0]  # Eliminate any extension
        filename = f"{filename_without_ext}.txt"  # Add .txt

    # If the PATH is not provided, the current directory (relative path) is used
    if path is None:
        path = os.getcwd()  # Get the current directory
    
    dir_path = os.path.join(path, filename)
        
    with open(dir_path, 'w') as file:
        # Introduce the parameters of the Image object first, in scientific notation with 8 decimals
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

def ReadParametersImageTXT(filename: str, path: str = None) -> clImage:
    """Function to read the parameters from a Parameters.txt file and return a clImage object, 
    including the matrix of the same size but zero, and the parameters of the atc dimensions"""
    # If a PATH is not delivered, we use the relative route (the current directory)
    if path is None:
        path = os.getcwd()  # Get the current directory
    # If you do not have the extension .txt, we add it
    if not filename.endswith(".txt"):
        filename += ".txt"
    # Build the full route of the file
    file_path = os.path.join(path, filename)

    params = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Read the first parameters until you reach the coordinates
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            
            # When we reach the line with the coordinates, we stop reading the parameters
            if line.startswith("X["):
                index += 1  # We jumped the coordinate heading line (x, y, z)
                break

            # Process the key parameter line: value
            if ":" in line:
                key, value = line.split(":")
                params[key.strip()] = value.strip()
            index += 1
        
        # Read the coordinates (x, y, z) and rebuild the matrix
        lenpxx = int(params.get("lenpxx", 256))
        lenpxy = int(params.get("lenpxy", 256))
        # Create an empty matrix with the right size
        matriz = np.zeros((lenpxx, lenpxy))
                   
    # Create the climage object with the parameters and the rebuilt matrix
    img = clImage(
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
        matriz=matriz
    )
    return img

def SaveImageToXYZ(image_obj, filename: str = None, path: str = None):
    """
   Saves an array in XYZ format with the numerical values ​​in scientific notation with 8 decimal places.
    
    Parameters:
    - image_obj: object of the clImage class that contains the array and parameters.
    - filename: the name of the file. If None, the channel name is used as the file name.
    - path: optional, the path where the file will be saved. If None, the relative path is used.
    """
    # If you are None, we use the channel as the file name
    if filename is None:
        filename = f"{image_obj.channel}.xyz"
    else:
        filename_without_ext = os.path.splitext(filename)[0]  # Eliminate any extension
        filename = f"{filename_without_ext}.xyz"  # Add .xyz

    # If the PATH is not provided, the current directory (relative path) is used
    if path is None:
        path = os.getcwd()  # Get the current directory
    
    dir_path = os.path.join(path, filename)
    
    # Calculate the steps on the x and y axes
    step_x = image_obj.size_x / image_obj.matriz.shape[0]  # Total size in x divided by the number of pixels in x
    step_y = image_obj.size_y / image_obj.matriz.shape[1]  # Total size and divided by the number of pixels in and
    
    with open(dir_path, 'w') as file:
        file.write("WSxM file copyright RMS\n")
        file.write("WSxM ASCII XYZ file\n")
        # Introduce the parameters of the Image object first, in scientific notation with 8 decimals
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
        
        # Add coordinates and units
        file.write(f"X[{image_obj.unitxy}]\tY[{image_obj.unitxy}]\tZ[{image_obj.unitz}]\n")
        
        # Write physical coordinates and matrix values ​​in scientific format
        for i in range(image_obj.matriz.shape[0]):
            for j in range(image_obj.matriz.shape[1]):
                # We convert the indices to physical coordinates
                x_real = i * step_x
                y_real = j * step_y
                file.write(f"{x_real:.8e} {y_real:.8e} {image_obj.matriz[i, j]:.8e}\n")

def LoadXYZ_FileToImage(filename: str, path: str = None) -> clImage:
    """Function to read the parameters of an XYZ txt file and return a clImage object, 
    including the matrix, and the parameters of the atc dimensions"""
    # If a PATH is not delivered, we use the relative route (the current directory)
    if path is None:
        path = os.getcwd()  # Get the current directory
    # If you do not have the extension .txt, we add it
    if not filename.endswith(".xyz"):
        filename += ".xyz"
    # Build the full route of the file
    file_path = os.path.join(path, filename)

    params = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Read the first parameters until you reach the coordinates
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            
            # When we reach the line with the coordinates, we stop reading the parameters
            if line.startswith("X["):
                index += 1  # We jumped the coordinate heading line (x, y, z)
                break

            # Process the key parameter line: value
            if ":" in line:
                key, value = line.split(":")
                params[key.strip()] = value.strip()
            index += 1
        
        # Read the coordinates (x, y, z) and rebuild the matrix
        lenpxx = int(params.get("len X", 256))
        lenpxy = int(params.get("len Y", 256))
        # Create an empty matrix with the right size
        matriz = np.zeros((lenpxx, lenpxy))
        for i in range(lenpxx):
            for j in range(lenpxy):
               
                z_value = float(lines[index].strip().split()[2])
                matriz[i, j] = z_value  # Assign the value of Z in the corresponding position
                index += 1  # Go to the next line in the file

           
    # Create the climage object with the parameters and the rebuilt matrix
    img = clImage(
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
        matriz=matriz
    )

    return img

def LoadChannelGWY_FileToImage(filename: str, Namechannel: ChannelType = 'Backward - Topography', path: str = None) -> clImage:
    """
   Function to load a Gwyddion (GWY) file and return a clImage object.
    
    Parameters:
    - filename: Name of the GWY file.
    - Namechannel: Specific channel of the GWY file to load.
    - path: Path where the file is located. If None, the current path is used.
    
    Returns:
    - A clImage object with the channel and image information.
    """
    
    # If a PATH is not provided, use the relative route (current directory)
    if path is None:
        path = os.getcwd()  # Get the current directory
    
    # If the file name does not have the extension '.gwy', we add it
    if not filename.endswith(".gwy"):
        filename += ".gwy"
    
    # Build the complete route to the file
    file_path = os.path.join(path, filename)
    
    # Verify if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
    
    # Load the Gwyddion file in memory
    obj = gwyfile.load(file_path)
    
    # Get the Datafield Objects of the Archive
    channels = gwyfile.util.get_datafields(obj)
    
    # Load the specific channel
    if Namechannel not in channels:
        raise ValueError(f"El canal '{Namechannel}' no se encuentra en el archivo.")
    
    channel = channels[Namechannel]
    
    # Extract the channel data matrix
    Img: np.ndarray = channel.data
    
    # Obtain the dimensions of the image
    lenpxx, lenpxy = Img.shape
    
    # Create the climage object with the channel and matrix parameters
    return clImage(
        channel=Namechannel,
        path=path,
        filename=filename,
        size_x=channel.xreal,
        size_y=channel.yreal,
        unitxy=channel['si_unit_xy'].unitstr,
        unitz=channel['si_unit_z'].unitstr,
        offset_x=channel.xoff,
        offset_y=channel.yoff,
        lenpxx=lenpxx,
        lenpxy=lenpxy,
        tag='',
        matriz=Img
    )

def ObtenerArchivosOrdenadosPorNumero(Directory: str, Exttype: str = '.gwy') -> List[str]:
    # Function to extract the number of the Filename_Etiqueta format _#. Extension
    def obtener_indice(filename: str) -> int:
        try:
            # We assume that the name follows the Filename_Etiqueta format _#. Extension
            return int(filename.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            # If the format is not expected, we return a high value to ignore it
            print(f"Advertencia: El archivo '{filename}' no sigue el formato esperado.")
            return float('inf')
    
    # Obtain all files with the specified extension
    archivos = [f for f in os.listdir(Directory) if f.endswith(Exttype)]
    
    # Order the files by the number extracted from the name
    archivos_ordenados = sorted(archivos, key=lambda x: obtener_indice(x))
    
    # Filter files that did not meet the expected format (where to obtain_indice returned inf)
    archivos_ordenados = [f for f in archivos_ordenados if obtener_indice(f) != float('inf')]
    
    return archivos_ordenados

def LoadAllImageFile_fromDirectory(Directory = None, Exttype: ExtentionType = '.gwy') -> List[clImage] :
    if Directory is None:
        Directory = OpenFolderDialog()
    
    # Obtaining valid and ordered file names
    filename_list = ObtenerArchivosOrdenadosPorNumero(Directory, Exttype)
    
    # Tour all files with specified extension and load Climage objects
    Image_list = []  # List to store loaded data

    # Load the images
    # Obtain and order the files in the directory according to the numerical index
    for filename in filename_list:
        if filename.endswith(Exttype):
            # Load the file * using the existing function
            if Exttype == '.gwy':
                img = LoadChannelGWY_FileToImage(filename,path = Directory)
            elif Exttype == '.xyz':
                img = LoadXYZ_FileToImage(filename,Directory)

            Image_list.append(img)

    return Image_list

def NormalizeNumpysOrclImages(all_matrices, nstd=4):

    # If there is not a list, we turn it into a single element list
    if isinstance(all_matrices, (np.ndarray, clImage)):  
        all_matrices = [all_matrices]
    
    # Extract matrices if the elements are climate objects
    matrices = [obj.matriz if isinstance(obj, clImage) else obj for obj in all_matrices]

    # Calculate the global average and standard deviation
    global_mean_value = np.mean([np.mean(matrix) for matrix in matrices])
    global_std_value = np.std([value for matrix in matrices for value in matrix])

    # Define the limits of standardization (± n standard deviations around average)
    vmin_global = global_mean_value - nstd * global_std_value
    vmax_global = global_mean_value + nstd * global_std_value

    # Calculate the real minimum and maximum among all matrices
    min_all_matrices = min(np.min(matrix) for matrix in matrices)
    max_all_matrices = max(np.max(matrix) for matrix in matrices)

    # Adjust Vmin and Vmax according to the limits defined by the data
    vmin = max(min_all_matrices, vmin_global)
    vmax = min(max_all_matrices, vmax_global)

    # Use the calculated range for standardization
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    return norm

def Default_cmap():
 # Function that returns the CMAP Classic of Topo
    # Control points for each channel, with the inverted values
    blue_points = np.array([[0, 0], [85, 0], [160, 76], [220, 213], [255, 251]])
    green_points = np.array([[0, 0], [17, 0], [47, 21], [106, 96], [182, 208], [232, 248], [255, 249]])
    red_points = np.array([[0, 0], [32, 119], [94, 198], [152, 238], [188, 249], [255, 249]])

    # Interpolation function for each channel
    def interpolate_channel(control_points):
        x = control_points[:, 0]
        y = control_points[:, 1]
        return interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Generate interpolated values ​​for 256 color levels
    def generate_colormap():
        levels = np.linspace(0, 255, 256)
        
        blue_interp = interpolate_channel(blue_points)
        green_interp = interpolate_channel(green_points)
        red_interp = interpolate_channel(red_points)
        
        blue_values = blue_interp(levels) / 255.0
        green_values = green_interp(levels) / 255.0
        red_values = red_interp(levels) / 255.0
        
        # Create an interpolated list
        colors = np.stack([red_values, green_values, blue_values], axis=1)
        return ListedColormap(colors)

    # Create the personalized color
    custom_cmap = generate_colormap()
    return custom_cmap

def SaveNumpyToPNG_By_PIL(data_matrix, output_dir, filename, norm = None, nstd = 4, colormap='copper'):
    # I normalize using the normal object normalize according to NSTD
    if norm is  None:
        norm =NormalizeNumpysOrclImages(data_matrix, nstd)
        
    norm_data = norm(data_matrix)
    # Get Matplotlib colormap
    cmap = plt.get_cmap(colormap)

    # Apply the colormap and convert it to an RGB image
    colored_data = cmap(norm_data)

    # 'Colored_data' is an RGBA matrix, we turn to Uint8 for Pillow (RGB)
    colored_data_uint8 = (colored_data[:, :, :3] * 255).astype(np.uint8)  # We ignore the Alfa Channel

    # Create the image from the NUMPY matrix with the colormap applied
    img = Image.fromarray(colored_data_uint8)

    # Verify if you have an extension and delete it, then add .png
    filename_without_ext = os.path.splitext(filename)[0]  # Eliminate any extension
    filename = f"{filename_without_ext}.png"  # Add .png
    
    # Save the image in the output folder
    output_filename = os.path.join(output_dir, filename)
    img.save(output_filename)

def DisplayImage_with_scale(image_obj: clImage, title: str = None, color_map: str = 'copper', std_devs: float = 4):
    """
   Function to display the image of a clImage object with color scale.
    The axes will have the size corresponding to size_x and size_y.
    
    Parameters:
    - image_obj: The clImage object containing the image and parameters.
    - color_map: Color map to use (e.g. 'gray', 'viridis', etc.).
    - std_devs: Number of standard deviations to limit the color scale.
                If None, the full range of the image will be used.
    """
    
    # Get the image (matrix)
    image_data = image_obj.matriz
    
    # Calculate the limits of the color scale depending on the standard deviation
    if std_devs is not None:
        mean_val = np.mean(image_data)
        std_val = np.std(image_data)
        vmin = mean_val - std_devs * std_val
        vmax = mean_val + std_devs * std_val
    else:
        vmin = np.min(image_data)
        vmax = np.max(image_data)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust the size as necessary
    
    # Show the image with the axes adjusted to the physical dimensions size_x and size_
    im = ax.imshow(image_data, cmap=color_map, vmin=vmin, vmax=vmax,
                   extent=[0, image_obj.size_x, 0, image_obj.size_y])
    
    # Add title with the channel or some other relevant information
    if title is None:
        ax.set_title(f"{image_obj.channel} - {image_obj.tag}")
    else:
        ax.set_title(f"{title}")
    
    # Add tags on the axes according to the units
    ax.set_xlabel(f"X [{image_obj.unitxy}]")
    ax.set_ylabel(f"Y [{image_obj.unitxy}]")
    
    # Add a color bar
    fig.colorbar(im, ax=ax, label=f"Z [{image_obj.unitz}]")
    
    # Show the image
    plt.show(block=False)

if __name__ == '__main__':


    
    # # Example of use
    # Matrix = NP.random.rand (256, 256)
    # image_obj = climage (channel = "ch1", size_x = 10.0, size_y = 10.0, unitxy = "µm", unitz = "nm", matrix = matrix)

    # # Save without specifying fillename (the channel is used as the file name)
    # Saveimagetoxyz_txt (image_obj)

    # IMG = Loadxyz_txtfiletoimage ('Ch1.txt')

    # Displayimage_with_scale (IMG)
    list = LoadAllImageFile_fromDirectory(Directory = None, Exttype='.gwy')
    DisplayImage_with_scale(list)
    plt.pause(0)
    # File = OpenFiledialog (default_path = r'c: \ users \ z78 \ documents \ captures', ext_file = [("txtfile", "*.txt")])
    # print ('open file:', file)

    # File = Savefiledialog ()
    # print ('Save faithful:', file)
    # IMGGWY = LoadChannelgwyfile ('Biofilm_10%_3.gwy', "Backward - Topography", 'Data')

    # Displayimage_with_scale (imggwy, std_devs = 10)

    # Savenumpytopng_by_pil (imggwy.matriz, imggwy.path, imggwy.filename, norm = none, nstd = 4, colormap = default_cmap ()))
    # PLT.Pause (0) # infinite pause to keep the window open
