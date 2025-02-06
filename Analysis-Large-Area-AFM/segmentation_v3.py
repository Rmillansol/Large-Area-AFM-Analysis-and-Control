"""
Author: Ruben Millan-Solsona
License: MIT

Function module to perform segmentation according to an already trained Yolo model.
"""
import cv2
import os
import numpy as np
from torchvision.ops import nms
from skimage import exposure
import torch
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import skimage.measure
import matplotlib.patches as patches
import h5py
import torch


def dividir_imagen_en_cuadrados_con_solapamiento(imagen, npx, solapamiento=0):
    """
    Divide una imagen en regiones cuadradas de tamaño npx. Si la imagen no se puede dividir
    perfectamente, se rellenan los píxeles faltantes con ceros. Se puede añadir un porcentaje 
    de solapamiento entre los cuadros.

    :param imagen: numpy array, la imagen de entrada.
    :param npx: int, el tamaño de cada región cuadrada en píxeles.
    :param solapamiento: float, porcentaje de solapamiento entre regiones (0 significa sin solapamiento).
    :return: lista de numpy arrays, cada uno representando una región cuadrada de npx por npx.
    """
    # Image dimensions
    height, width = imagen.shape[:2]

    # Calculate the step (displacement) between pictures taking into account the overlap
    paso = int(npx * (1 - solapamiento))

    # Calculate the number of necessary frames throughout each axis
    
    n_cuadros_x = (width ) // paso + 1 if width > npx else 1
    n_cuadros_y = (height) // paso + 1 if height > npx else 1

    # Create a larger image with zeros if necessary to complete the division
    padded_width = (n_cuadros_x - 1) * paso + npx
    padded_height = (n_cuadros_y - 1) * paso + npx
    imagen_padded = np.zeros((padded_height, padded_width) + imagen.shape[2:], dtype=imagen.dtype)
    imagen_padded[:height, :width] = imagen  # Copy the original image to the padded

    # Create the list of square images
    lista_imagenes = []
    lista_pos = []
    for y in range(0, padded_height - npx + 1, paso):
        for x in range(0, padded_width - npx + 1, paso):
            cuadro = imagen_padded[y:y+npx, x:x+npx]
            lista_imagenes.append(cuadro)
            lista_pos.append([x,y])

    return lista_imagenes, lista_pos, n_cuadros_x, n_cuadros_y, padded_height, padded_width

def mostrar_subimagenes(lista_imagenes, n_cuadros_x, n_cuadros_y):
    """
    Muestra las subimágenes en un array de subplots según su posición original.
    
    :param lista_imagenes: lista de numpy arrays, imágenes cuadradas que forman la imagen original.
    :param n_cuadros_x: int, número de cuadros a lo largo del eje x (ancho).
    :param n_cuadros_y: int, número de cuadros a lo largo del eje y (alto).
    :param npx: int, tamaño de cada imagen cuadrada (npx x npx).
    """
    fig, axs = plt.subplots(n_cuadros_y, n_cuadros_x, figsize=(15, 15))

    for idx, img in enumerate(lista_imagenes):
        fila = idx // n_cuadros_x
        col = idx % n_cuadros_x
        axs[fila, col].imshow(img, cmap='gray')
        axs[fila, col].axis('off')

    plt.show()

def SaveMasksListInhdf5(masks_list, filepath):
    with h5py.File(filepath, 'w') as f:
        idx=0
        for idx, masks in enumerate(masks_list):
            for mask in masks:
                f.create_dataset(f"mask_{idx}", data=mask)
                idx+=1
    print(f"Máscaras guardadas en {filepath}")

def LoadMasksFile(indx):
    """
    Carga todas las máscaras densas desde un único archivo .npz.

    :param directorio_base: Ruta del directorio base donde está el archivo.
    :param indx: Índice que se utilizó para nombrar el archivo.
    :return: Lista de máscaras densas (NumPy arrays).
    """
    # Define the file route
    filename = os.path.join("Masks", f"Masks_{indx}.npz")
    
    # Load the .npz file
    data = np.load(filename)
    
    # Extract masks as a list of arrays
    masks_list = [data[key] for key in data]
    print(f"Máscaras cargadas desde {filename}")
    
    return masks_list

def calcular_bounding_box(mask):
    # Find the positions where the mask is 1
    posiciones = np.where(mask > 0)

    # If there are no points in the mask, None returns
    if posiciones[0].size == 0:
        return None

    # Calculate the Bounding Box: [Xmin, Ymin, Xmax, Ymax] as an array of Numpy
    bounding_box = np.array([posiciones[1].min(), posiciones[0].min(), posiciones[1].max(), posiciones[0].max()], dtype=np.float32)
    
    return bounding_box

def IsMaskInEdges(mask, patch_size, tolerance=1):
    # Define edges with tolerance
    borde_superior = slice(0, tolerance)  # The first 'tolerance' rows
    borde_inferior = slice(patch_size - tolerance, patch_size)  # The latest 'tolerance' rows
    borde_izquierdo = slice(0, tolerance)  # The first 'tolerance' columns
    borde_derecho = slice(patch_size - tolerance, patch_size)  # The latest 'tolerance' columns

    # Check if the mask touches any of the edges within tolerance
    toca_borde = (
        np.any(mask[borde_superior, :]) or  # Tolerance on the upper edge
        np.any(mask[borde_inferior, :]) or  # Tolerance on the lower edge
        np.any(mask[:, borde_izquierdo]) or  # Tolerance on the left edge
        np.any(mask[:, borde_derecho])       # Right edge tolerance
    )
    
    return toca_borde

def SmoothMask(mask,sigma = 8, above_area_threshold = 4000, below_area_threshold = 50 ):
    
    # Find the regions connected in the mask
    labels = skimage.measure.label(mask)
    props = skimage.measure.regionprops(labels)

    if len(props) > 0:      # If there are regions
        # Select the largest connected region
        largest_region = max(props, key=lambda x: x.area)

        # Create a new mask with the largest region
        largest_mask = np.zeros_like(mask)
        largest_mask[labels == largest_region.label] = 1

        # Apply Gaussian soft -mask to the mask
        smoothed_mask = gaussian_filter(largest_mask.astype(float), sigma=sigma)
        smoothed_mask=smoothed_mask > 0.5
        # Calculate the softened mask area
        area = np.sum(smoothed_mask)  

        # Filter by area threshold
        if area < below_area_threshold or area > above_area_threshold:
          return None, None
        
        box = calcular_bounding_box(smoothed_mask)
        return smoothed_mask, box
    else:
        return None, None
          
def PredictImagesList(fileh5_path, image_list, lista_pos,
                       scale, model, sigma = 8,  above_area_threshold = 4000, below_area_threshold = 50, conf=0.02, equalize = True, is_INTER_LANCZOS4 = True):
    # Function predicts a list of images
    boxes_list = []
    scores_list = []
    index_list = []
    pos_list =  []

    patch_size=640
    with h5py.File(fileh5_path, 'w') as fh5:
   
        # Perform predictions in each subimagen
        idx=0
        for subimagen,(pos_x, pos_y) in zip(image_list,lista_pos):
            # Preparation Image for Yolo
            image_tensor = PrepareImageforYOLO(subimagen,equalize, is_INTER_LANCZOS4)
            
             # Perform the prediction
            result = model.predict(image_tensor, conf=conf)

            if result is None or len(result) == 0:
                print("No se detectaron objetos en la imagen.")
            else:
                # Get the masks (if any)
                masks = result[0].masks.data.cpu().numpy() if result[0].masks is not None else None
                # Obtain the boxes with 6 elements (x1, y1, x2, y2, conf, class)
                boxes_with_conf_class = result[0].boxes.data.cpu().numpy()  # (X1, Y1, X2, Y2, Conf, Class)

                # Extract only coordinates (X1, Y1, X2, Y2)
                boxes = boxes_with_conf_class[:, :4]

                # Extract the scores (Confidence)
                scores = boxes_with_conf_class[:, 4]
                # Boxes, scores, masks = predict_y_nms (image_tensor, model, conf = conf)

                # Adjust the coordinates of the boxes to the global reference system
                pos_x_new = pos_x * scale
                pos_y_new = pos_y * scale

                if masks is None or boxes is None or scores is None:
                    print("Error: Una de las variables (masks, boxes o scores) es None.")
                else:

                    # Maskaras and Ajusto Box filter
                    
                    for mask,box_old,score in zip(masks,boxes,scores):
                        try:
                            toca_borde = IsMaskInEdges(mask,patch_size,5)
                            if not toca_borde:
                                # If the edge does not touch the mask I save the mask already leaked
                                smooth_mask,box = SmoothMask(mask,sigma = sigma, above_area_threshold = above_area_threshold, below_area_threshold = below_area_threshold )
                                if smooth_mask is not None:
                                    # Modify the positions x1, x2, y1, y2 of the boxes to global coordinates
                                    box[[0, 2]] += pos_x_new        # Adjust X1 and X2
                                    box[[1, 3]] += pos_y_new        # Adjust Y1 and Y2

                                    
                                    fh5.create_dataset(f"mask_{idx}", data=smooth_mask)
                                    pos_list.append([pos_x_new, pos_y_new])
                                    boxes_list.append(box)
                                    scores_list.append(score)
                                    index_list.append(idx)
                                    idx +=1                         
                        except TypeError as e:
                            print(f"Error de tipo en la predicción (iteración {idx}): {e}")
                        except Exception as e:
                            print(f"Error general en la predicción (iteración {idx}): {e}")
                
        fh5.create_dataset(f"scores", data=np.array(scores_list))
        fh5.create_dataset(f"boxes", data=np.array(boxes_list))
        fh5.create_dataset(f"pos", data=np.array(pos_list))
        fh5.create_dataset(f"indexs", data=np.array(index_list))

    return boxes_list, scores_list, pos_list

def filtrar_y_guardar_mascaras(prediction_path,prediction_path_out, keep_idx):
    """
    Filtra las máscaras de un archivo HDF5 usando una lista de índices y las guarda en un nuevo archivo
    renombradas en orden consecutivo.
    
    :param filepath_original: Ruta del archivo HDF5 original.
    :param filepath_nuevo: Ruta del archivo HDF5 donde se guardarán las máscaras filtradas.
    :param keep_idx: Lista de índices de las máscaras que se desean mantener.
    """
    with h5py.File(prediction_path, 'r') as archivo_original:
        with h5py.File(prediction_path, 'w') as archivo_nuevo:
            # Iterate on Keep_idx and copy the selected masks to the new file
            for nuevo_idx, idx in enumerate(keep_idx):
                # Load the original file mask using the index
                mascara = archivo_original[f"mask_{idx}"][:]
                
                # Save the mask in the new renowned file with New_idx in consecutive order
                archivo_nuevo.create_dataset(f"mask_{nuevo_idx}", data=mascara)
                print(f"Máscara {idx} guardada como mask_{nuevo_idx} en el nuevo archivo.")
    
    print(f"Todas las máscaras seleccionadas se han guardado en {prediction_path_out}.")

def apply_nms_list(prediction_path,prediction_path_out,boxes_list, scores_list, lista_pos, iou_threshold=0.5):
    """
    Aplica Non-Maximum Suppression (NMS) a las predicciones de YOLOv8.

    :param boxes_list: Lista de cajas de las predicciones.
    :param scores_list: Lista de puntuaciones.
    :param lista_pos: Lista de posiciones globales de cada fragmento.
    :param npx: Tamaño de los fragmentos (por defecto 640x640).
    :param iou_threshold: Umbral de IoU para aplicar el NMS.
    :return: Cajas, puntuaciones, máscaras filtradas y las posiciones correspondientes a las máscaras.
    """
    global_boxes = []
    global_scores = []

    # Recover data
    with h5py.File(prediction_path, 'r') as fh5:
        # Load the data
        scores_list = fh5["scores"][:]
        boxes_list = fh5["boxes"][:]
        lista_pos = fh5["pos"][:]
    
        # Convert accumulated lists
        global_boxes = torch.tensor(boxes_list)  # Convert boxes list into a tensioner
        global_scores = torch.tensor(scores_list)  # Convert tensioning scores

        # Apply NMS globally
        keep_idx = nms(global_boxes, global_scores, iou_threshold)

        # Filter the boxes and scores using the NMS indices
        filtered_boxes = global_boxes[keep_idx].numpy()
        filtered_scores = global_scores[keep_idx].numpy()

        # Convert Keep_idx to an array numpy to use it in masks and positions
        keep_idx_np = keep_idx.numpy()

        # Filter positions using the indices
        filtered_pos = [lista_pos[i] for i in keep_idx_np]  # Keep only the corresponding positions

        with h5py.File(prediction_path_out, 'w') as fh5new:
            for nuevo_idx, idx in enumerate(keep_idx):
                # Load the original file mask using the index
                mask= fh5[f"mask_{idx}"][:]
                # Save the mask in the new renowned file with New_idx in consecutive order
                fh5new.create_dataset(f"mask_{nuevo_idx}", data=mask)
                print(f"Máscara {idx} guardada como mask_{nuevo_idx} en el nuevo archivo.")
        
            fh5new.create_dataset(f"scores", data=np.array(filtered_scores))
            fh5new.create_dataset(f"boxes", data=np.array(filtered_boxes))
            fh5new.create_dataset(f"pos", data=np.array(filtered_pos))


    return filtered_boxes, filtered_scores, filtered_pos

def combinar_mascaras_en_imagen(path_masks, lista_pos, img_height, img_width):
    """
    Combina las máscaras individuales de diferentes objetos en una sola imagen global utilizando la posición global de cada conjunto de máscaras.

    :param masks: Lista de listas de máscaras filtradas (después de NMS).
    :param lista_pos: Lista de posiciones globales de las máscaras.
    :param img_height: Altura de la imagen global.
    :param img_width: Anchura de la imagen global.
    :param npx: Tamaño de las máscaras individuales (por defecto 640x640).
    :return: Imagen con todas las máscaras combinadas.
    """
    # Create an empty global mask of the original image size
    global_mask = np.zeros((img_height, img_width))

    with h5py.File(path_masks, 'r') as fih5:
        
        # Iterate on each list of masks and their corresponding position
        for idx, (pos_x, pos_y) in enumerate(lista_pos):
            pos_x = int(pos_x)  # Convert post_x to whole
            pos_y = int(pos_y)  # Convert post_ to whole

            mask=fih5[f"mask_{idx}"][:]
            # Insert the individual mask into its position in the global image
            # We use np.maximum so as not to overwrite masks that already exist
            global_mask[pos_y:pos_y + 640, pos_x:pos_x + 640] = np.maximum(
                            global_mask[pos_y:pos_y + 640, pos_x:pos_x + 640], mask)

            # global_mask [pos_y: pos_y + 640, post_x: post_x + 640] + = mask

    return global_mask

def filtrar_mascaras(mask_path,boxes, scores, posiciones, area_threshold, sigma=1):
    """
    Filtra las máscaras seleccionando la región conectada más grande, aplicando suavizado gaussiano,
    y eliminando máscaras cuyo área sea menor que un umbral. También filtra las posiciones correspondientes.

    :param masks: Lista de máscaras a filtrar.
    :param boxes: Lista de cajas (boxes) correspondientes a las máscaras.
    :param scores: Lista de puntuaciones correspondientes a las máscaras.
    :param posiciones: Lista de posiciones globales de las máscaras.
    :param area_threshold: Umbral mínimo de área para mantener una máscara.
    :param sigma: Valor de sigma para el suavizado gaussiano (por defecto 1).
    :return: Máscaras, cajas, puntuaciones y posiciones filtradas.
    """
    keep_idx = []
    filtered_boxes = []
    filtered_scores = []
    filtered_posiciones = []
    with h5py.File(mask_path, 'r+') as f:  # Open in Reading/Writing mode

        for idx,(box, score, pos) in enumerate(zip(boxes, scores, posiciones)):
            mask = f[f"mask_{idx}"][:]
            # Find the regions connected in the mask
            labels = skimage.measure.label(mask)
            props = skimage.measure.regionprops(labels)

            if len(props) == 0:
                continue  # If there are no regions, go to the next mask

            # Select the largest connected region
            largest_region = max(props, key=lambda x: x.area)

            # Create a new mask with the largest region
            largest_mask = np.zeros_like(mask)
            largest_mask[labels == largest_region.label] = 1

            # Apply Gaussian soft -mask to the mask
            smoothed_mask = gaussian_filter(largest_mask.astype(float), sigma=sigma)

            # Calculate the softened mask area
            area = np.sum(smoothed_mask > 0.5)  # Consider pixels where the mask is greater than 0.5

            # Filter by area threshold
            if area >= area_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_posiciones.append(pos)  # Maintain the corresponding position
                keep_idx.append(idx)        # Mask to save
                
                # Eliminate existing dataset
                del f[f"mask_{idx}"]
                # Create the new dataset with the same name
                f.create_dataset(f"mask_{idx}", data=smoothed_mask > 0.5)  # Keep only areas where the value is greater than 0.5)

    filtrar_y_guardar_mascaras('masks_nms.h5', 'masks_nms_filter.h5', keep_idx)

    return np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_posiciones)

def dibujar_boxes_con_imagen(boxes, img_height, img_width, filtered_pos, imagen=None):
    """
    Dibuja las cajas de predicción en una imagen.

    :param boxes: Lista de cajas (boxes) filtradas después de NMS. Deben tener formato [x1, y1, x2, y2].
    :param img_height: Altura de la imagen sobre la cual se dibujarán las cajas.
    :param img_width: Anchura de la imagen sobre la cual se dibujarán las cajas.
    :param filtered_pos: Lista de posiciones globales de las máscaras/cajas.
    :param imagen: Imagen de fondo para dibujar las cajas (opcional). Si es None, se creará una imagen en blanco.
    """
    # If an image is not provided, we create a blank image (black)
    if imagen is None:
        imagen = np.zeros((img_height, img_width), dtype=np.uint8)

    # Create a figure and an axis
    fig, ax = plt.subplots(1,figsize=[10,10])

    # Show the image
    ax.imshow(imagen, cmap='gray')

    # Draw each box
    for (box, (pos_x, pos_y)) in zip(boxes, filtered_pos):
        x1, y1, x2, y2 = box  # Box coordinates

        # Create a rectangle with the coordinates of the box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the axis
        ax.add_patch(rect)

    # Show the image with the boxes
    plt.show()

def show_image(img,cmap = 'viridis'):
    """
    Muestra una sola imagen.

    :param img: numpy array, imagen a mostrar
    """
    plt.figure(figsize=(5, 5), dpi=120)  # Adjust the size of the image window if necessary
    plt.imshow(img, cmap)  # You can adjust 'CMAP' according to the image type (E.G., 'Gray', 'Viridis', etc.)
    plt.axis('off')  # Remove the axes
    plt.show()

def reconstruir_imagen(lista_imagenes, lista_pos, height, width, npx):
    """
    Reconstruye la imagen original a partir de una lista de imágenes cuadradas teniendo en cuenta el solapamiento.

    :param lista_imagenes: lista de numpy arrays, imágenes cuadradas que forman la imagen original.
    :param lista_pos: lista de posiciones [pos_x, pos_y]  de los recortes de la imagen original.
    :param height: Altura de la imagen original
    :param with: Altura de la imagen original
    :param npx: int, tamaño de cada imagen cuadrada (npx x npx).
    :return: numpy array, la imagen reconstruida.
    """
    # Obtain matrix format
    shape = lista_imagenes[0].shape
    # Get the type of image data
    dtype = lista_imagenes[0].dtype
    # I think the resulting image with the same type of data as the list
    if len(shape)==3:
        n,m,k = shape
        exist_3channel= True
        imagen_reconstruida = np.zeros((height + npx, width + npx, k), dtype=dtype)
    else:
        exist_3channel=False
        imagen_reconstruida = np.zeros((height + npx, width + npx), dtype=dtype)
    
    contador_superposiciones = np.zeros_like(imagen_reconstruida)  # To control overlapping areas
    for img, [pos_x, pos_y] in zip(lista_imagenes,lista_pos):
        # Add the current image to the reconstructed image in the right position
        if exist_3channel:
            # Add the current image to the reconstructed image in the right position
            imagen_reconstruida[pos_y:pos_y + npx, pos_x:pos_x + npx,:] += img
            # Maintain a record of how many times each pixel overlaps
            contador_superposiciones[pos_y:pos_y + npx, pos_x:pos_x + npx,:] += 1            
        else:
            # Add the current image to the reconstructed image in the right position
            imagen_reconstruida[pos_y:pos_y + npx, pos_x:pos_x + npx] += img
            # Maintain a record of how many times each pixel overlaps
            contador_superposiciones[pos_y:pos_y + npx, pos_x:pos_x + npx] += 1

    # Divide the image by the overlapping counter to average the overlapping areas
    imagen_reconstruida = np.divide(imagen_reconstruida, contador_superposiciones, where=contador_superposiciones > 0)

    imagen_out = imagen_reconstruida[0:height, 0:width,:]     # I trim the image to the original size
    return imagen_out

def PrepareImageforYOLO(img,equalize = True, is_INTER_LANCZOS4 = True):
   
    if is_INTER_LANCZOS4:
        image_resized = cv2.resize(img, (640, 640),interpolation=cv2.INTER_LANCZOS4)
    else:
        image_resized = cv2.resize(img, (640, 640),interpolation=cv2.INTER_LINEAR)
    if equalize:
        img_eq = exposure.equalize_hist(image_resized)
        image_tensor = torch.from_numpy(img_eq).permute(2, 0, 1).float().unsqueeze(0)
    else:
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0)
    return image_tensor

def predict_y_nms(image_tensor, model, conf=0.02,iou_threshold=0.5):
    # Function predicts an image with Yolo
    # Perform the prediction
    result = model.predict(image_tensor, conf=conf)
    # Get the masks (if any)
    boxes_nms,scores_nms,masks_nms=apply_nms(result, iou_threshold)

    return boxes_nms,scores_nms,masks_nms

def apply_nms(result, iou_threshold=0.5):
    """
    Aplica Non-Maximum Suppression (NMS) a las predicciones de YOLOv8.

    :param result_list: Lista de resultados con las predicciones de segmentación.
    :param iou_threshold: Umbral de IoU para aplicar el NMS (por defecto 0.5).
    :return: Lista de resultados con NMS aplicado.
    """
    
    # Get the masks (if any)
    masks = result[0].masks.data.cpu().numpy() if result[0].masks is not None else None
    # Obtain the boxes with 6 elements (x1, y1, x2, y2, conf, class)
    boxes_with_conf_class = result[0].boxes.data.cpu().numpy()  # (X1, Y1, X2, Y2, Conf, Class)

    # Extract only coordinates (X1, Y1, X2, Y2)
    boxes = boxes_with_conf_class[:, :4]

    # Extract the scores (Confidence)
    scores = boxes_with_conf_class[:, 4]

    # Accumulate boxes, scores and masks globally and turn to tensioners to apply NMS
    global_boxes=torch.tensor(boxes)
    global_scores=torch.tensor(scores)
    global_masks=torch.tensor(masks)

    # Apply NMS globally
    keep_idx = nms(global_boxes, global_scores, iou_threshold)

    # Filter the boxes, scores and masks using the NMS indices
    filtered_boxes = global_boxes[keep_idx].numpy()
    filtered_scores = global_scores[keep_idx].numpy()
    filtered_masks = global_masks[keep_idx].numpy()

    return filtered_boxes,filtered_scores,filtered_masks

def combine_masks(masks):
    # Initialize Blank Combined Mask
    combined_mask = np.zeros_like(masks[0])

    # Combine All Individual Masks Into One Image
    for i in range(masks.shape[0]):
        combined_mask = np.maximum(combined_mask, masks[i])
        # combined_mask += masks [i]
    
    return combined_mask

def PredictList(image_list, model,conf = 0.02):
    # Function that segments a whole list of images and returns a list of results
    result_list = []
    boxes = []
    scores = []
    masks = []
    for image in image_list:
        # Resize the image at 640x640 as expected by YOLOV8-SEG
        image_resized = cv2.resize(image, (640, 640))

        # Convert the image to the format expected by YOLO (C, H, W and as Pytorch tensor)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0)

        # Execute the inference using the Yolov8-SEG model
        results = model.predict(image_tensor, conf=conf)

        # Get the masks (if any)
        masks_t = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
        # Obtain the boxes with 6 elements (x1, y1, x2, y2, conf, class)
        boxes_with_conf_class = results[0].boxes.data.cpu().numpy()  # (X1, Y1, X2, Y2, Conf, Class)

        # Extract only coordinates (X1, Y1, X2, Y2)
        boxes_t = boxes_with_conf_class[:, :4]

        # Extract the scores (Confidence)
        scores_t = boxes_with_conf_class[:, 4]
        
        # Accumulate boxes, scores and masks globally and turn to tensioners to apply NMS
        boxes.append(boxes_t)
        scores.append(scores_t)
        masks.append(masks_t)
    
    return boxes,scores,masks

def CalcularPropiedadesYGuardar(scores,masks_path, output_file):
    """
    Calcula propiedades geométricas de las bacterias segmentadas y guarda los resultados en un archivo de texto.

    :param scores: Lista con los scores de los objetos
    :param masks: Lista con las mascaras de cada objeto detectado
    :param out_file: Ruta o nombre del archivo

    """
    # Column names
    propiedades = ['Bacteria_Num',  'Score', 'Area', 'Centroid_X', 'Centroid_Y', 'Eccentricity',
                   'Extent', 'Perimeter', 'Orientation', 'Major_Axis_Length', 'Minor_Axis_Length',
                   'Solidity']
    
    with h5py.File(masks_path, 'r') as f:  # Open in Reading/Writing mode

        with open(output_file, 'w') as file:
            # Write the names of the columns in the front row
            file.write('\t'.join(propiedades) + '\n')

            # Iterate about each result (each image)
            count_cells=0
            for idx, score in enumerate(scores):
                mask = f[f"mask_{idx}"][:]
                labeled_mask = label(mask)  # Label the regions connected in the mask
                props = regionprops(labeled_mask)

                # Calculate the properties for each region (bacteria)
                for prop in props:
                    count_cells+=1

                    area = prop.area
                    centroid_x, centroid_y = prop.centroid
                    eccentricity = prop.eccentricity
                    extent = prop.extent
                    perimeter = prop.perimeter
                    orientation = prop.orientation
                    major_axis = prop.major_axis_length
                    minor_axis = prop.minor_axis_length
                    solidity = prop.solidity

                    # Write the properties in the file, separated by tabs
                    file.write(f"{count_cells}\t{score}\t{area}\t{centroid_x}\t{centroid_y}\t{eccentricity}\t"
                            f"{extent}\t{perimeter}\t{orientation}\t{major_axis}\t{minor_axis}\t"
                            f"{solidity}\n")                

        print(f"Datos guardados en {output_file}")
        print(f"# of bacteria: {count_cells} ")

def DoPropertyMapAndTXT(prediction_path, OutTXTFile_path, img_height, img_width):
    """
    Crea un mapa de propiedades para un conjunto de máscaras. Cada propiedad se asigna a una capa separada
    de la imagen global de salida y guarda una archivo de texto con las propiedades ademas de guardar un nuevo
    grupo en el archivo h5 con los mapas.

    :param prediction_path: Ruta al archivo h5 con los datos de la prediccion
    :param OutTXTFile_path: Ruta archivo de texto para guardar una tabla con todas las propiedades de cada bacteria
    :param img_height: Altura de la imagen global.
    :param img_width: Anchura de la imagen global.
    :return: Un diccionario con mapas de propiedades para cada máscara.
    """
    
    # Column and maps names
    propiedades = ['Bacteria_Num',  'Score', 'Area', 'Centroid_X', 'Centroid_Y', 'Eccentricity',
                   'Extent', 'Perimeter', 'Orientation', 'Major_Axis_Length', 'Minor_Axis_Length',
                   'Solidity']

    # Create a dictionary to store property maps
    prop_maps = {prop: np.full((img_height, img_width), np.nan) for prop in propiedades}

    with h5py.File(prediction_path, 'a') as fh5:  # Open in Reading/Writing mode
        # Load the data
        scores_list = fh5["scores"][:]
        boxes_list = fh5["boxes"][:]
        pos_list = fh5["pos"][:]
        with open(OutTXTFile_path, 'w') as file:
            # Write the names of the columns in the front row
            file.write('\t'.join(propiedades) + '\n')

            # Iterate on each mask, position and corresponding score
            for idx, ((pos_x, pos_y), score, box) in enumerate(zip(pos_list, scores_list, boxes_list)):
                pos_x = int(pos_x)  # Convert post_x to whole
                pos_y = int(pos_y)  # Convert post_ to whole

                mask = fh5[f"mask_{idx}"][:]
                labeled_mask = label(mask)  # Label the regions connected in the mask
                props = regionprops(labeled_mask)

                # Calculate the properties for each region connected
                for prop in props:
                    area = prop.area
                    eccentricity = prop.eccentricity
                    extent = prop.extent
                    perimeter = prop.perimeter
                    orientation = prop.orientation
                    major_axis = prop.major_axis_length
                    minor_axis = prop.minor_axis_length
                    solidity = prop.solidity
                # Calculation of bacteria positions
                    centro_x = (box[0] + box[2]) / 2
                    centro_y = (box[1] + box[3]) / 2
                # Write the properties in the text file, separated by tabs
                    file.write(f"{idx}\t{score}\t{area}\t{centro_x}\t{centro_y}\t{eccentricity}\t"
                            f"{extent}\t{perimeter}\t{orientation}\t{major_axis}\t{minor_axis}\t"
                            f"{solidity}\n")   
                    
                # Map the properties in global positions where the mask is 1
                    mask_indices = mask == 1  # Create a Boolean array where the mask is 1
                    # Bacteria_num
                    prop_maps['Bacteria_Num'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = idx

                    # Score
                    prop_maps['Score'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = score

                    # Area
                    prop_maps['Area'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = area

                    # Centroid_x
                    prop_maps['Centroid_X'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = pos_x

                     # Centroid_
                    prop_maps['Centroid_Y'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = pos_y

                    # ECCENTRICITY
                    prop_maps['Eccentricity'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = eccentricity

                    # Extensive
                    prop_maps['Extent'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = extent

                    # Perimeter
                    prop_maps['Perimeter'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = perimeter

                    # Orientation
                    prop_maps['Orientation'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = orientation

                    # Major axis
                    prop_maps['Major_Axis_Length'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = major_axis

                    # Minor axis
                    prop_maps['Minor_Axis_Length'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = minor_axis

                    # Solidity
                    prop_maps['Solidity'][pos_y:pos_y + mask.shape[0], pos_x:pos_x + mask.shape[1]][mask_indices] = solidity
            # I think a new group I keep the maps
            grp = fh5.create_group('Maps')
            # Loop to save each matrix with its key as the name of the dataset
            for key, matriz in prop_maps.items():
                # Save each matrix with the name of the key
                grp.create_dataset(key, data=matriz)
                
    return prop_maps
