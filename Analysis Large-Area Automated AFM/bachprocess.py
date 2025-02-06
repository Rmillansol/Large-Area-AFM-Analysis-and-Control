import os
import copy
from typing import List, Tuple
from AFMclasses import clImage, ChannelType, ExtentionType
import managefiles as mgf
import flattening_v2 as fla

def PrepareStichByAutoFlattenPlus(Directory=None, 
                                  Exttype: ExtentionType = '.gwy',
                                  Autosave=True, 
                                  perc=0.4, nstd=4) -> Tuple[List[clImage], List[clImage]]:
    """
    Processes images in a directory by applying an automatic flattening technique. Each image is flattened,
    masked, and optionally saved. The function also performs global normalization.

    :param Directory: Directory containing the images.
    :param Exttype: File extension type to look for.
    :param Autosave: If True, saves the processed images.
    :param perc: Percentage parameter for flattening.
    :param nstd: Number of standard deviations for normalization.
    :return: Tuple of lists containing flattened images and masks.
    """
    
    Imgs = mgf.LoadAllImageFile_fromDirectory(Directory, Exttype)
    
    if Autosave:
        out_folder = mgf.DoFolderWithDate('Flatten', Imgs[0].path)
        Mask_path = os.path.join(out_folder, 'Mask_XYZ')
        os.makedirs(Mask_path, exist_ok=True)
    
    # Save a text file with the parameters of the first image; assumed to be the same for all images.
    mgf.SaveParmaImageTotxt(Imgs[0], filename='imageparameters.txt', path=out_folder)
    
    # Lists to store flattened images and masks
    flattened_images = []
    masks = []
    
    # Iterate over images to apply flattening and create masks
    for Img in Imgs:
        # Make a deep copy of the image object
        Img_flatten = copy.deepcopy(Img)
        Img_mask = copy.deepcopy(Img)
        matriz = Img.matriz
        
        # Apply AutoFlattenPlus function to obtain the flattened image and mask
        flatten, mask = fla.AutoFlattenPlus(matriz, perc)
        
        # Update the matrix attributes with the results
        Img_flatten.matriz = flatten
        Img_mask.matriz = mask
        
        if Autosave:
            # Set file names and save paths for flattened images and masks
            Img_flatten.filename = 'flatten_' + Img.filename
            Img_flatten.path = out_folder
            Img_mask.filename = 'mask_' + Img.filename
            Img_mask.path = Mask_path

            # Save images in XYZ format
            mgf.SaveImageToXYZ(Img_flatten, Img_flatten.filename, out_folder)
            mgf.SaveImageToXYZ(Img_mask, Img_mask.filename, Img_mask.path)
        
        # Append the results to the lists
        flattened_images.append(Img_flatten)
        masks.append(Img_mask)

    # Create a normalization object for global normalization of all images
    norm = mgf.NormalizeNumpysOrclImages(flattened_images, nstd)  # Values for global normalization
    print('vmax:', norm.vmax)
    print('vmin:', norm.vmin)
   
    if Autosave:
        # Save each normalized flattened image as PNG
        for Img in flattened_images:
            mgf.SaveNumpyToPNG_By_PIL(Img.matriz, out_folder, Img.filename, norm=norm, nstd=nstd)
                            
    # Return both lists
    return flattened_images, masks

def PrepareStichByManualPlaneByPoints(Directory=None, 
                                      Exttype: ExtentionType = '.gwy',
                                      Autosave=True, nstd=4) -> Tuple[List[clImage], List[clImage]]:
    """
    Processes images in a directory by manually adjusting the plane using selected points. Each image is flattened,
    masked, and optionally saved. The function also performs global normalization.

    :param Directory: Directory containing the images.
    :param Exttype: File extension type to look for.
    :param Autosave: If True, saves the processed images.
    :param nstd: Number of standard deviations for normalization.
    :return: Tuple of lists containing flattened images and masks.
    """
    
    Imgs = mgf.LoadAllImageFile_fromDirectory(Directory, Exttype)
    
    if Autosave:
        out_folder = mgf.DoFolderWithDate('Flatten', Imgs[0].path)
        Mask_path = os.path.join(out_folder, 'Mask_XYZ')
        os.makedirs(Mask_path, exist_ok=True)
    
    # Save a text file with the parameters of the first image; assumed to be the same for all images.
    mgf.SaveParmaImageTotxt(Imgs[0], filename='imageparameters.txt', path=out_folder)
    
    # Lists to store flattened images and masks
    flattened_images = []
    masks = []
    
    # Iterate over images to manually adjust the plane and create masks
    for Img in Imgs:
        # Make a deep copy of the image object
        Img_flatten = copy.deepcopy(Img)
        Img_mask = copy.deepcopy(Img)
        matriz = Img.matriz
        
        # Apply manual plane adjustment function to obtain the flattened image and mask
        flatten, mask = fla.SubtracGlobalPlaneManualByPoints(matriz, show=True)
        
        # Update the matrix attributes with the results
        Img_flatten.matriz = flatten
        Img_mask.matriz = mask
        
        if Autosave:
            # Set file names and save paths for flattened images and masks
            Img_flatten.filename = 'flatten_' + Img.filename
            Img_flatten.path = out_folder
            Img_mask.filename = 'mask_' + Img.filename
            Img_mask.path = Mask_path

            # Save images in XYZ format
            mgf.SaveImageToXYZ(Img_flatten, Img_flatten.filename, out_folder)
            mgf.SaveImageToXYZ(Img_mask, Img_mask.filename, Img_mask.path)
        
        # Append the results to the lists
        flattened_images.append(Img_flatten)
        masks.append(Img_mask)

    # Create a normalization object for global normalization of all images
    norm = mgf.NormalizeNumpysOrclImages(flattened_images, nstd)  # Values for global normalization
    print('vmax:', norm.vmax)
    print('vmin:', norm.vmin)
   
    if Autosave:
        # Save each normalized flattened image as PNG
        for Img in flattened_images:
            mgf.SaveNumpyToPNG_By_PIL(Img.matriz, out_folder, Img.filename, norm=norm, nstd=nstd)
                            
    # Return both lists
    return flattened_images, masks

if __name__ == '__main__':
    # Run the manual plane adjustment function
    PrepareStichByManualPlaneByPoints(nstd=2)
