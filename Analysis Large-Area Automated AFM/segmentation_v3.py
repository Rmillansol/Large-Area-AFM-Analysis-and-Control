"""
Author: Ruben Millan-Solsona
Date of Creation: August 2024

Description:
This module contains functions for image segmentation using a pre-trained YOLO model, including image division 
with overlap, YOLO prediction handling, non-maximum suppression (NMS), mask saving/loading, and calculating 
properties for segmented regions. The processed segmentation data can also be stored and retrieved efficiently.

Dependencies:
- os
- numpy
- cv2
- torch
- h5py
- matplotlib.pyplot
- skimage (exposure, measure)
- torchvision.ops (nms)
- scipy.ndimage (gaussian_filter)
"""

import os
import cv2
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from torchvision.ops import nms
from skimage import exposure
from skimage.measure import regionprops, label
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches


def dividir_imagen_en_cuadrados_con_solapamiento(imagen, npx, solapamiento=0):
    """
    Divide an image into overlapping square regions of size npx. If the image does not divide perfectly,
    missing pixels are filled with zeros. Allows optional overlap.

    :param imagen: numpy array, the input image.
    :param npx: int, size of each square region in pixels.
    :param solapamiento: float, overlap percentage between regions (0 = no overlap).
    :return: list of numpy arrays (sub-images), original positions, grid dimensions, padded height, and width.
    """
    height, width = imagen.shape[:2]
    paso = int(npx * (1 - solapamiento))
    n_cuadros_x = (width) // paso + 1 if width > npx else 1
    n_cuadros_y = (height) // paso + 1 if height > npx else 1
    padded_width = (n_cuadros_x - 1) * paso + npx
    padded_height = (n_cuadros_y - 1) * paso + npx
    imagen_padded = np.zeros((padded_height, padded_width) + imagen.shape[2:], dtype=imagen.dtype)
    imagen_padded[:height, :width] = imagen

    lista_imagenes, lista_pos = [], []
    for y in range(0, padded_height - npx + 1, paso):
        for x in range(0, padded_width - npx + 1, paso):
            cuadro = imagen_padded[y:y + npx, x:x + npx]
            lista_imagenes.append(cuadro)
            lista_pos.append([x, y])

    return lista_imagenes, lista_pos, n_cuadros_x, n_cuadros_y, padded_height, padded_width


def mostrar_subimagenes(lista_imagenes, n_cuadros_x, n_cuadros_y):
    """
    Displays sub-images in a grid of subplots according to their original position.

    :param lista_imagenes: list of numpy arrays, square images forming the original image.
    :param n_cuadros_x: int, number of squares along the x-axis.
    :param n_cuadros_y: int, number of squares along the y-axis.
    """
    fig, axs = plt.subplots(n_cuadros_y, n_cuadros_x, figsize=(15, 15))
    for idx, img in enumerate(lista_imagenes):
        fila, col = idx // n_cuadros_x, idx % n_cuadros_x
        axs[fila, col].imshow(img, cmap='gray')
        axs[fila, col].axis('off')
    plt.show()


def SaveMasksListInhdf5(masks_list, filepath):
    """
    Saves a list of masks to an HDF5 file.

    :param masks_list: List of numpy arrays representing masks.
    :param filepath: Path to save the HDF5 file.
    """
    with h5py.File(filepath, 'w') as f:
        idx = 0
        for masks in masks_list:
            for mask in masks:
                f.create_dataset(f"mask_{idx}", data=mask)
                idx += 1
    print(f"Masks saved to {filepath}")


def LoadMasksFile(indx):
    """
    Loads all dense masks from a single .npz file.

    :param indx: Index used for naming the .npz file.
    :return: List of dense masks as numpy arrays.
    """
    filename = os.path.join("Masks", f"Masks_{indx}.npz")
    data = np.load(filename)
    masks_list = [data[key] for key in data]
    print(f"Masks loaded from {filename}")
    return masks_list


def calcular_bounding_box(mask):
    """
    Calculates the bounding box of a mask where mask > 0.

    :param mask: numpy array, binary mask.
    :return: numpy array, bounding box as [xmin, ymin, xmax, ymax].
    """
    posiciones = np.where(mask > 0)
    if posiciones[0].size == 0:
        return None
    bounding_box = np.array([posiciones[1].min(), posiciones[0].min(), posiciones[1].max(), posiciones[0].max()], dtype=np.float32)
    return bounding_box


def eliminar_mascaras_borde(prediction_path, patch_size):
    """
    Removes masks that touch the border in an HDF5 file.

    :param prediction_path: Path to the HDF5 file.
    :param patch_size: Size of the patch used for masking.
    :return: Lists of filtered boxes, scores, and positions.
    """
    boxes_filtradas, scores_filtradas, pos_filtradas = [], [], []
    bordes_superior, bordes_inferior = 0, patch_size - 1
    bordes_izquierdo, bordes_derecho = 0, patch_size - 1
    nuevo_idx = 0

    with h5py.File(prediction_path, 'r+') as fh5:
        for idx, (box, score, pos) in enumerate(zip(boxes_list, scores_list, pos_list)):
            mask = fh5[f"mask_{idx}"][:]
            toca_borde = (
                np.any(mask[bordes_superior, :]) or
                np.any(mask[bordes_inferior, :]) or
                np.any(mask[:, bordes_izquierdo]) or
                np.any(mask[:, bordes_derecho])
            )
            if not toca_borde:
                fs.create_dataset(f"mask_{nuevo_idx}", data=mask)
                boxes_filtradas.append(box)
                scores_filtradas.append(score)
                pos_filtradas.append(pos)
                nuevo_idx += 1

    return boxes_filtradas, scores_filtradas, pos_filtradas


def IsMaskInEdges(mask, patch_size, tolerance=1):
    """
    Checks if the mask touches the edges within a certain tolerance.

    :param mask: numpy array, binary mask.
    :param patch_size: int, size of the patch.
    :param tolerance: int, tolerance range to check the edges.
    :return: Boolean, True if mask touches any edge, else False.
    """
    borde_superior, borde_inferior = slice(0, tolerance), slice(patch_size - tolerance, patch_size)
    borde_izquierdo, borde_derecho = slice(0, tolerance), slice(patch_size - tolerance, patch_size)
    toca_borde = (
        np.any(mask[borde_superior, :]) or
        np.any(mask[borde_inferior, :]) or
        np.any(mask[:, borde_izquierdo]) or
        np.any(mask[:, borde_derecho])
    )
    return toca_borde


def SmoothMask(mask, sigma=8, above_area_threshold=4000, below_area_threshold=50):
    """
    Smooths the largest connected region in a mask using Gaussian filtering. Masks with areas outside
    specified thresholds are removed.

    :param mask: numpy array, input binary mask.
    :param sigma: int, sigma value for Gaussian smoothing.
    :param above_area_threshold: int, upper area threshold to filter large masks.
    :param below_area_threshold: int, lower area threshold to filter small masks.
    :return: Smoothed mask and bounding box.
    """
    labels = skimage.measure.label(mask)
    props = skimage.measure.regionprops(labels)
    if props:
        largest_region = max(props, key=lambda x: x.area)
        largest_mask = np.zeros_like(mask)
        largest_mask[labels == largest_region.label] = 1
        smoothed_mask = gaussian_filter(largest_mask.astype(float), sigma=sigma) > 0.5
        area = np.sum(smoothed_mask)
        if below_area_threshold < area < above_area_threshold:
            return smoothed_mask, calcular_bounding_box(smoothed_mask)
    return None, None


def PredictImagesList(fileh5_path, image_list, lista_pos, scale, model, sigma=8, above_area_threshold=4000, below_area_threshold=50, conf=0.02, equalize=True, is_INTER_LANCZOS4=True):
    """
    Performs segmentation predictions on a list of images, saving results and masks to an HDF5 file.

    :param fileh5_path: Path to the output HDF5 file.
    :param image_list: List of images to segment.
    :param lista_pos: List of positions for each segment.
    :param scale: Scaling factor for adjusting global coordinates.
    :param model: Pre-trained YOLO model.
    :param sigma: Gaussian smoothing parameter.
    :param above_area_threshold: Maximum area threshold for a mask.
    :param below_area_threshold: Minimum area threshold for a mask.
    :param conf: Confidence threshold for the model.
    :param equalize: Boolean, True if histogram equalization is applied to images.
    :param is_INTER_LANCZOS4: Boolean, True if using INTER_LANCZOS4 interpolation.
    :return: Lists of boxes, scores, and positions after NMS.
    """
    boxes_list, scores_list, index_list, pos_list = [], [], [], []
    patch_size = 640
    with h5py.File(fileh5_path, 'w') as fh5:
        idx = 0
        for subimagen, (pos_x, pos_y) in zip(image_list, lista_pos):
            image_tensor = PrepareImageforYOLO(subimagen, equalize, is_INTER_LANCZOS4)
            result = model.predict(image_tensor, conf=conf)

            if result is None or len(result) == 0:
                print("No objects detected.")
            else:
                masks = result[0].masks.data.cpu().numpy() if result[0].masks is not None else None
                boxes_with_conf_class = result[0].boxes.data.cpu().numpy()
                boxes, scores = boxes_with_conf_class[:, :4], boxes_with_conf_class[:, 4]
                pos_x_new, pos_y_new = pos_x * scale, pos_y * scale

                if masks is not None:
                    for mask, box_old, score in zip(masks, boxes, scores):
                        try:
                            toca_borde = IsMaskInEdges(mask, patch_size, 5)
                            if not toca_borde:
                                smooth_mask, box = SmoothMask(mask, sigma, above_area_threshold, below_area_threshold)
                                if smooth_mask is not None:
                                    box[[0, 2]] += pos_x_new
                                    box[[1, 3]] += pos_y_new
                                    fh5.create_dataset(f"mask_{idx}", data=smooth_mask)
                                    pos_list.append([pos_x_new, pos_y_new])
                                    boxes_list.append(box)
                                    scores_list.append(score)
                                    index_list.append(idx)
                                    idx += 1
                        except TypeError as e:
                            print(f"Type error during prediction (iteration {idx}): {e}")
                        except Exception as e:
                            print(f"General error during prediction (iteration {idx}): {e}")

        fh5.create_dataset("scores", data=np.array(scores_list))
        fh5.create_dataset("boxes", data=np.array(boxes_list))
        fh5.create_dataset("pos", data=np.array(pos_list))
        fh5.create_dataset("indexs", data=np.array(index_list))

    return boxes_list, scores_list, pos_list

import h5py
import torch
import numpy as np
from skimage.measure import regionprops, label
from torchvision.ops import nms
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import exposure
import cv2

def filtrar_y_guardar_mascaras(prediction_path, prediction_path_out, keep_idx):
    """
    Filters masks from an HDF5 file using a list of indices and saves them in a new file with consecutive renaming.

    :param prediction_path: Path of the original HDF5 file.
    :param prediction_path_out: Path where the filtered HDF5 file will be saved.
    :param keep_idx: List of indices of masks to keep.
    """
    with h5py.File(prediction_path, 'r') as archivo_original:
        with h5py.File(prediction_path_out, 'w') as archivo_nuevo:
            for nuevo_idx, idx in enumerate(keep_idx):
                mascara = archivo_original[f"mask_{idx}"][:]
                archivo_nuevo.create_dataset(f"mask_{nuevo_idx}", data=mascara)
                print(f"Mask {idx} saved as mask_{nuevo_idx} in the new file.")
    print(f"All selected masks saved in {prediction_path_out}.")


def apply_nms_list(prediction_path, prediction_path_out, boxes_list, scores_list, lista_pos, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) on YOLOv8 predictions, filtering boxes, scores, and masks.

    :param prediction_path: Path of the input HDF5 file.
    :param prediction_path_out: Path where the output HDF5 file with filtered results will be saved.
    :param boxes_list: List of bounding boxes.
    :param scores_list: List of confidence scores.
    :param lista_pos: List of global positions for each fragment.
    :param iou_threshold: IoU threshold for applying NMS.
    :return: Filtered boxes, scores, and positions.
    """
    with h5py.File(prediction_path, 'r') as fh5:
        scores_list = fh5["scores"][:]
        boxes_list = fh5["boxes"][:]
        lista_pos = fh5["pos"][:]

        global_boxes = torch.tensor(boxes_list)
        global_scores = torch.tensor(scores_list)
        keep_idx = nms(global_boxes, global_scores, iou_threshold)
        filtered_boxes = global_boxes[keep_idx].numpy()
        filtered_scores = global_scores[keep_idx].numpy()
        keep_idx_np = keep_idx.numpy()
        filtered_pos = [lista_pos[i] for i in keep_idx_np]

        with h5py.File(prediction_path_out, 'w') as fh5new:
            for nuevo_idx, idx in enumerate(keep_idx):
                mask = fh5[f"mask_{idx}"][:]
                fh5new.create_dataset(f"mask_{nuevo_idx}", data=mask)
                print(f"Mask {idx} saved as mask_{nuevo_idx} in the new file.")
            fh5new.create_dataset("scores", data=np.array(filtered_scores))
            fh5new.create_dataset("boxes", data=np.array(filtered_boxes))
            fh5new.create_dataset("pos", data=np.array(filtered_pos))

    return filtered_boxes, filtered_scores, filtered_pos


def combinar_mascaras_en_imagen(path_masks, lista_pos, img_height, img_width):
    """
    Combines individual object masks into a single global image using the global position of each mask.

    :param path_masks: Path to the HDF5 file containing masks.
    :param lista_pos: List of global positions for each mask.
    :param img_height: Height of the global image.
    :param img_width: Width of the global image.
    :return: Combined global mask image.
    """
    global_mask = np.zeros((img_height, img_width))
    with h5py.File(path_masks, 'r') as fih5:
        for idx, (pos_x, pos_y) in enumerate(lista_pos):
            pos_x, pos_y = int(pos_x), int(pos_y)
            mask = fih5[f"mask_{idx}"][:]
            global_mask[pos_y:pos_y + 640, pos_x:pos_x + 640] = np.maximum(global_mask[pos_y:pos_y + 640, pos_x:pos_x + 640], mask)
    return global_mask


def filtrar_mascaras(mask_path, boxes, scores, posiciones, area_threshold, sigma=1):
    """
    Filters masks, selecting the largest connected region, applying Gaussian smoothing, and discarding masks below an area threshold.

    :param mask_path: Path to the HDF5 file containing masks.
    :param boxes: List of bounding boxes.
    :param scores: List of confidence scores.
    :param posiciones: List of global positions for each mask.
    :param area_threshold: Minimum area threshold for keeping a mask.
    :param sigma: Gaussian smoothing factor.
    :return: Filtered boxes, scores, and positions.
    """
    keep_idx, filtered_boxes, filtered_scores, filtered_posiciones = [], [], [], []
    with h5py.File(mask_path, 'r+') as f:
        for idx, (box, score, pos) in enumerate(zip(boxes, scores, posiciones)):
            mask = f[f"mask_{idx}"][:]
            labels = label(mask)
            props = regionprops(labels)
            if len(props) == 0:
                continue
            largest_region = max(props, key=lambda x: x.area)
            largest_mask = np.zeros_like(mask)
            largest_mask[labels == largest_region.label] = 1
            smoothed_mask = gaussian_filter(largest_mask.astype(float), sigma=sigma) > 0.5
            area = np.sum(smoothed_mask)
            if area >= area_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_posiciones.append(pos)
                keep_idx.append(idx)
                del f[f"mask_{idx}"]
                f.create_dataset(f"mask_{idx}", data=smoothed_mask)

    filtrar_y_guardar_mascaras(mask_path, 'filtered_masks.h5', keep_idx)
    return np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_posiciones)


def dibujar_boxes_con_imagen(boxes, img_height, img_width, filtered_pos, imagen=None):
    """
    Draws prediction boxes on an image.

    :param boxes: List of bounding boxes filtered after NMS, format [x1, y1, x2, y2].
    :param img_height: Height of the image.
    :param img_width: Width of the image.
    :param filtered_pos: List of global positions for the masks/boxes.
    :param imagen: Background image (optional), creates a blank image if None.
    """
    if imagen is None:
        imagen = np.zeros((img_height, img_width), dtype=np.uint8)
    fig, ax = plt.subplots(1, figsize=[10, 10])
    ax.imshow(imagen, cmap='gray')
    for box, (pos_x, pos_y) in zip(boxes, filtered_pos):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def show_image(img, cmap='viridis'):
    """
    Displays a single image.

    :param img: numpy array, the image to display.
    :param cmap: Color map for display.
    """
    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.show()

def reconstruir_imagen(lista_imagenes, lista_pos, height, width, npx):
    """
    Reconstructs the original image from a list of square images, accounting for overlap.

    :param lista_imagenes: List of numpy arrays, square images that form the original image.
    :param lista_pos: List of positions [pos_x, pos_y] of the cropped sections in the original image.
    :param height: Height of the original image.
    :param width: Width of the original image.
    :param npx: int, size of each square image (npx x npx).
    :return: numpy array, the reconstructed image.
    """
    # Determine the data type and dimensions of the images
    shape = lista_imagenes[0].shape
    dtype = lista_imagenes[0].dtype
    exist_3channel = len(shape) == 3

    # Create the reconstructed image and an overlap counter
    if exist_3channel:
        _, _, k = shape
        imagen_reconstruida = np.zeros((height + npx, width + npx, k), dtype=dtype)
    else:
        imagen_reconstruida = np.zeros((height + npx, width + npx), dtype=dtype)

    contador_superposiciones = np.zeros_like(imagen_reconstruida)  

    for img, [pos_x, pos_y] in zip(lista_imagenes, lista_pos):
        if exist_3channel:
            imagen_reconstruida[pos_y:pos_y + npx, pos_x:pos_x + npx, :] += img
            contador_superposiciones[pos_y:pos_y + npx, pos_x:pos_x + npx, :] += 1
        else:
            imagen_reconstruida[pos_y:pos_y + npx, pos_x:pos_x + npx] += img
            contador_superposiciones[pos_y:pos_y + npx, pos_x:pos_x + npx] += 1

    imagen_reconstruida = np.divide(imagen_reconstruida, contador_superposiciones, where=contador_superposiciones > 0)
    return imagen_reconstruida[0:height, 0:width, :] if exist_3channel else imagen_reconstruida[0:height, 0:width]


def PrepareImageforYOLO(img, equalize=True, is_INTER_LANCZOS4=True):
    """
    Prepares an image for use with the YOLO model.

    :param img: numpy array, the image to process.
    :param equalize: bool, whether to apply histogram equalization.
    :param is_INTER_LANCZOS4: bool, uses INTER_LANCZOS4 if True; otherwise, uses INTER_LINEAR.
    :return: PyTorch tensor, resized and optionally equalized image.
    """
    interpolation = cv2.INTER_LANCZOS4 if is_INTER_LANCZOS4 else cv2.INTER_LINEAR
    image_resized = cv2.resize(img, (640, 640), interpolation=interpolation)
    if equalize:
        img_eq = exposure.equalize_hist(image_resized)
        return torch.from_numpy(img_eq).permute(2, 0, 1).float().unsqueeze(0)
    else:
        return torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0)


def predict_y_nms(image_tensor, model, conf=0.02, iou_threshold=0.5):
    """
    Makes a prediction on an image using YOLO and applies Non-Maximum Suppression (NMS).

    :param image_tensor: PyTorch tensor, the processed image ready for YOLO.
    :param model: YOLO model for prediction.
    :param conf: float, confidence threshold for the prediction.
    :param iou_threshold: float, Intersection over Union threshold for NMS.
    :return: Filtered boxes, scores, and masks after applying NMS.
    """
    result = model.predict(image_tensor, conf=conf)
    boxes_nms, scores_nms, masks_nms = apply_nms(result, iou_threshold)
    return boxes_nms, scores_nms, masks_nms


def apply_nms(result, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to YOLOv8 predictions.

    :param result: List of segmentation prediction results.
    :param iou_threshold: float, IoU threshold for NMS (default 0.5).
    :return: Filtered boxes, scores, and masks after NMS.
    """
    # Extract masks if they exist
    masks = result[0].masks.data.cpu().numpy() if result[0].masks is not None else None
    # Extract boxes with confidence and class data (x1, y1, x2, y2, conf, class)
    boxes_with_conf_class = result[0].boxes.data.cpu().numpy()

    # Only retrieve coordinates (x1, y1, x2, y2)
    boxes = boxes_with_conf_class[:, :4]
    # Retrieve scores (confidence)
    scores = boxes_with_conf_class[:, 4]

    # Convert lists of boxes, scores, and masks to tensors for NMS
    global_boxes = torch.tensor(boxes)
    global_scores = torch.tensor(scores)
    global_masks = torch.tensor(masks)

    # Apply global NMS
    keep_idx = nms(global_boxes, global_scores, iou_threshold)

    # Filter boxes, scores, and masks using NMS indices
    filtered_boxes = global_boxes[keep_idx].numpy()
    filtered_scores = global_scores[keep_idx].numpy()
    filtered_masks = global_masks[keep_idx].numpy()

    return filtered_boxes, filtered_scores, filtered_masks


def combine_masks(masks):
    """
    Combines multiple individual masks into a single mask.

    :param masks: numpy array of individual masks to be combined.
    :return: Combined mask.
    """
    combined_mask = np.zeros_like(masks[0])

    for i in range(masks.shape[0]):
        combined_mask = np.maximum(combined_mask, masks[i])
    
    return combined_mask


def PredictList(image_list, model, conf=0.02):
    """
    Segments a list of images and returns a list of predictions.

    :param image_list: List of images to segment.
    :param model: YOLO model for prediction.
    :param conf: float, confidence threshold for predictions.
    :return: Lists of boxes, scores, and masks.
    """
    boxes, scores, masks = [], [], []

    for image in image_list:
        # Resize image to 640x640 as required by YOLOv8-seg
        image_resized = cv2.resize(image, (640, 640))
        # Convert image to format expected by YOLO (C, H, W and as PyTorch tensor)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0)

        # Run inference with YOLOv8-seg model
        results = model.predict(image_tensor, conf=conf)

        # Retrieve masks if they exist
        masks_t = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        # Retrieve boxes (x1, y1, x2, y2, conf, class)
        boxes_with_conf_class = results[0].boxes.data.cpu().numpy()

        # Only extract coordinates (x1, y1, x2, y2)
        boxes_t = boxes_with_conf_class[:, :4]
        # Extract scores (confidence)
        scores_t = boxes_with_conf_class[:, 4]
        
        # Accumulate boxes, scores, and masks globally for NMS
        boxes.append(boxes_t)
        scores.append(scores_t)
        masks.append(masks_t)
    
    return boxes, scores, masks
def DoPropertyMapAndTXT(prediction_path, OutTXTFile_path, img_height, img_width):
    """
    Creates a property map for a set of masks. Each property is assigned to a separate layer
    of the output global image and saves a text file with the properties, as well as a new
    group in the h5 file containing the maps.

    :param prediction_path: Path to the h5 file with prediction data.
    :param OutTXTFile_path: Path to the text file for saving a table with each bacteria's properties.
    :param img_height: Height of the global image.
    :param img_width: Width of the global image.
    :return: A dictionary with property maps for each mask.
    """
    
    # Column names and property map keys
    properties = ['Bacteria_Num', 'Score', 'Area', 'Centroid_X', 'Centroid_Y', 'Eccentricity',
                  'Extent', 'Perimeter', 'Orientation', 'Major_Axis_Length', 'Minor_Axis_Length', 'Solidity']

    # Initialize a dictionary to store property maps with NaN as a default value
    prop_maps = {prop: np.full((img_height, img_width), np.nan) for prop in properties}

    with h5py.File(prediction_path, 'a') as fh5:  # Open in read/write mode
        # Load data from the h5 file
        scores_list = fh5["scores"][:]
        boxes_list = fh5["boxes"][:]
        pos_list = fh5["pos"][:]

        with open(OutTXTFile_path, 'w') as file:
            # Write column headers in the first row
            file.write('\t'.join(properties) + '\n')

            # Iterate over each mask, position, and score
            for idx, ((pos_x, pos_y), score, box) in enumerate(zip(pos_list, scores_list, boxes_list)):
                pos_x, pos_y = int(pos_x), int(pos_y)  # Convert positions to integers

                # Retrieve mask from h5 file
                mask = fh5[f"mask_{idx}"][:]
                labeled_mask = label(mask)  # Label connected regions in the mask
                props = regionprops(labeled_mask)

                # Calculate properties for each connected region
                for prop in props:
                    area = prop.area
                    eccentricity = prop.eccentricity
                    extent = prop.extent
                    perimeter = prop.perimeter
                    orientation = prop.orientation
                    major_axis = prop.major_axis_length
                    minor_axis = prop.minor_axis_length
                    solidity = prop.solidity

                    # Calculate the center coordinates of the bacteria
                    centro_x = (box[0] + box[2]) / 2
                    centro_y = (box[1] + box[3]) / 2

                    # Write properties to the text file, separated by tabs
                    file.write(f"{idx}\t{score}\t{area}\t{centro_x}\t{centro_y}\t{eccentricity}\t"
                               f"{extent}\t{perimeter}\t{orientation}\t{major_axis}\t{minor_axis}\t"
                               f"{solidity}\n")   

                    # Map properties to the global positions where the mask is 1
                    mask_indices = mask == 1  # Boolean array where the mask is 1

                    # Assign each property to the appropriate location in prop_maps
                    prop_maps['Bacteria_Num'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = idx
                    prop_maps['Score'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = score
                    prop_maps['Area'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = area
                    prop_maps['Centroid_X'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = centro_x
                    prop_maps['Centroid_Y'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = centro_y
                    prop_maps['Eccentricity'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = eccentricity
                    prop_maps['Extent'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = extent
                    prop_maps['Perimeter'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = perimeter
                    prop_maps['Orientation'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = orientation
                    prop_maps['Major_Axis_Length'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = major_axis
                    prop_maps['Minor_Axis_Length'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = minor_axis
                    prop_maps['Solidity'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = solidity

            # Create a new group in the h5 file to store property maps
            grp = fh5.create_group('Maps')
            # Loop to save each matrix with its key as the dataset name
            for key, matrix in prop_maps.items():
                grp.create_dataset(key, data=matrix)

    return prop_maps
