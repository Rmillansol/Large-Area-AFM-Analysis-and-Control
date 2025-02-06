
"""
Author: Ruben Millan-Solsona
Date of Creation: August 2024

Description:
This module contains functions bachproces to prepare the files for stiching etc.

Dependencies:
- os
- copy
- typing
- AFMclasses (contains clImage, ChannelType, ExtentionType)
- managefiles (custom module for file management)
- flattening_v2

"""
import os
import copy
from typing import List, Tuple
from AFMclasses import clImage, ChannelType, ExtentionType
import managefiles as mgf
import flattening_v2 as fla

def PrepareStichByAutoFlattenPlus(Directory = None, 
                            Exttype: ExtentionType = '.gwy',
                            perc = 0.4, nstd =4 ) -> Tuple[List[clImage], List[clImage]] :
    
    Imgs = mgf.LoadAllImageFile_fromDirectory(Directory, Exttype)
    
   
    out_folder = mgf.DoFolderWithDate('Flatten', Imgs[0].path)
    Mask_path = os.path.join(out_folder, 'Mask_XYZ')
    os.makedirs(Mask_path, exist_ok=True)
    
    # Except for a TXT file with the parameters of the First Iamgen.They are supposed to be the same for all
    mgf.SaveParmaImageTotxt(Imgs[0],filename='imageparameters.txt', path=out_folder)
    # Ready to store flattened images and masks
    flattened_images = []
    masks = []
    
    # Iterate about flattene and mask images
    for Img in Imgs:
         # Make a deep copy of the IMG object
        Img_flatten = copy.deepcopy(Img)
        Img_mask = copy.deepcopy(Img)
        matriz = Img.matriz
        # Apply the autoflattenplus function to obtain the flattened image and the mask
        flatten, mask = fla.AutoFlattenPlus(matriz, perc)
        
        Img_flatten.matriz = flatten
        Img_mask.matriz = mask
        
        Img_flatten.filename = 'flatten_' + Img.filename
        Img_flatten.path = out_folder
        Img_mask.filename = 'mask_' + Img.filename
        Img_mask.path = Mask_path

        mgf.SaveImageToXYZ(Img_flatten,Img_flatten.filename,out_folder)
        mgf.SaveImageToXYZ(Img_mask,Img_mask.filename,Img_mask.path)
        
            
        # Add the results to the lists
        flattened_images.append(Img_flatten)
        masks.append(Img_mask)

    # Normal object to normalize all images globally
    norm = mgf.NormalizeNumpysOrclImages(flattened_images, nstd)        # Values ​​to normalize all images globally
  
    # Iterate about images
    for Img in flattened_images:        
        mgf.SaveNumpyToPNG_By_PIL(Img.matriz,out_folder,Img.filename,norm = norm, nstd = nstd)
                            
    # Return both lists
    return flattened_images, masks, out_folder

def PrepareStichByManualPlaneByPoints(Directory = None, 
                            Exttype: ExtentionType = '.gwy',
                            Autosave =True, nstd =4 ) -> Tuple[List[clImage], List[clImage]] :
    
    Imgs = mgf.LoadAllImageFile_fromDirectory(Directory, Exttype)
    
    if Autosave:
        out_folder = mgf.DoFolderWithDate('Flatten', Imgs[0].path)
        Mask_path = os.path.join(out_folder, 'Mask_XYZ')
        os.makedirs(Mask_path, exist_ok=True)
    
    # Except for a TXT file with the parameters of the First Iamgen.They are supposed to be the same for all
    mgf.SaveParmaImageTotxt(Imgs[0],filename='imageparameters.txt', path=out_folder)
    # Ready to store flattened images and masks
    flattened_images = []
    masks = []
    
    # Iterate about flattene and mask images
    for Img in Imgs:
         # Make a deep copy of the IMG object
        Img_flatten = copy.deepcopy(Img)
        Img_mask = copy.deepcopy(Img)
        matriz = Img.matriz
        # Apply the autoflattenplus function to obtain the flattened image and the mask
        flatten, mask  = fla.SubtracGlobalPlaneManualByPoints(matriz, show=True)
       
        Img_flatten.matriz = flatten
        Img_mask.matriz = mask
        if Autosave:
            Img_flatten.filename = 'flatten_' + Img.filename
            Img_flatten.path = out_folder
            Img_mask.filename = 'mask_' + Img.filename
            Img_mask.path = Mask_path

            mgf.SaveImageToXYZ(Img_flatten,Img_flatten.filename,out_folder)
            mgf.SaveImageToXYZ(Img_mask,Img_mask.filename,Img_mask.path)
           
            
        # Add the results to the lists
        flattened_images.append(Img_flatten)
        masks.append(Img_mask)

    # Normal object to normalize all images globally
    norm = mgf.NormalizeNumpysOrclImages(flattened_images, nstd)        # Values ​​to normalize all images globally
    print('vmax :', norm.vmax)
    print('vmin :', norm.vmin)
   
    if Autosave:
        # Iterate about images
        for Img in flattened_images:        
            if Autosave:
                mgf.SaveNumpyToPNG_By_PIL(Img.matriz,out_folder,Img.filename,norm = norm, nstd = nstd)
                            
    # Return both lists
    return flattened_images, masks


if __name__ == '__main__':
        
    # PREPARASTICHBYAUTOFLATTENPLUS (NSTD = 2, perc = 0.7)
    PrepareStichByManualPlaneByPoints(nstd = 2)