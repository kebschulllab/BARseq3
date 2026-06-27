"""
find cell masks, expand cell masks
map gene calls to cell masks
remove repeats in overlap
convert gene result dataframe into anndata
"""

import numpy as np
import pandas as pd
import anndata as ad
from cellpose import models
from cellpose import denoise, io
from skimage.segmentation import expand_labels
import copy
from datetime import datetime
# print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"")

# import cv2
# import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import multiprocessing as mp
from multiprocessing import Process, Pool
import warnings
warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import skimage as sk
# from skimage import io
# from skimage.color import rgb2gray
# from skimage.data import stereo_motorcycle, vortex
# from skimage.transform import warp
# from skimage.registration import optical_flow_tvl1, optical_flow_ilk
# from skimage.segmentation import expand_labels, watershed
# from skimage.color import label2rgb
# import IPython.display
# import SimpleITK as sitk
# import time, os, sys
# from urllib.parse import urlparse
# from cellpose import utils, io, plot
# from scipy.spatial import KDTree


def find_masks(dapi_images, nissl_images, cellpose_model, cellpose_channel, cellpose_diameter): #0-255 intensity values
    # prep images
    if (dapi_images is not None) and (nissl_images is not None):
        #Merge Nissl and DAPI images for cell segmentation
        # print(datetime.now().strftime(“%d/%m/%Y %H:%M:%S”), “Merging Nissl and DAPI images”)
        merged_images = []
        for i in range(len(dapi_images)): 
            merged_images.append(np.stack([np.zeros(dapi_images[i].shape), nissl_images[i]/np.max(nissl_images[i]), dapi_images[i]/np.max(dapi_images[i])], axis =2))
    elif (dapi_images is not None) and (nissl_images is None):
        merged_images = dapi_images
    elif (dapi_images is None) and (nissl_images is not None):
        merged_images = nissl_images

    all_masks = []
    if cellpose_model == 'cyto3':
        model = models.Cellpose(gpu=True, model_type=cellpose_model)
        for i in range(len(merged_images)): # zip(channels, file)
            masks, flows, styles, diams = model.eval(merged_images[i], diameter=cellpose_diameter, channels=cellpose_channel,flow_threshold=1.5,cellprob_threshold= -5, min_size=35) #,flow_threshold=1.5,cellprob_threshold= -5, min_size=20 from chatgpt
            all_masks.append(masks)
    else:
        model = models.CellposeModel(gpu=True, model_type=cellpose_model)
        for i in range(len(merged_images)): # zip(channels, file)
            masks, flows, styles = model.eval(merged_images[i], diameter=cellpose_diameter, channels=cellpose_channel,flow_threshold=1.5,cellprob_threshold= -5, min_size=35) #,flow_threshold=1.5,cellprob_threshold= -5, min_size=20 from chatgpt
            all_masks.append(masks)

    return all_masks

# Chatgpt version find_masks_better
from cellpose import models
import numpy as np

def _p01(x):
    # 稳健归一化到 [0,1]，避免极值影响
    lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
    if hi <= lo: hi = lo + 1.0
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def find_masks_better(
    dapi_images,                 # list[np.ndarray(H,W)] 或 (N,H,W)
    nissl_images=None,           # 仅保留形参以兼容旧调用，内部不用
    cellpose_model='nuclei',     # 先用更稳的 nuclei；后续再试 'tn3'
    cellpose_channel=(0,0),      # 灰度图用 (0,0) 或 [0,0]
    cellpose_diameter=24,        # 先用你之前可行的直径
    flow_threshold=0.2,          # 保守一点
    cellprob_threshold=0.0,      # 先别太激进（-3 容易全空）
    normalize='p99',             # 稳健归一化
    use_gpu=True                 # 有 GPU 就 True
):
    # 支持 list 或 (N,H,W)
    if isinstance(dapi_images, np.ndarray) and dapi_images.ndim == 3:
        imgs = [dapi_images[i] for i in range(dapi_images.shape[0])]
    else:
        imgs = list(dapi_images)

    # 创建与旧代码风格相同的模型（逐张 eval，返回四元组）
    model = models.Cellpose(gpu=use_gpu, model_type=str(cellpose_model))

    masks_all = []
    for idx, im in enumerate(imgs):
        im = im.astype(np.float32)
        im_norm = _p01(im) if normalize == 'p99' else (im / (im.max() + 1e-8))
        im_u8 = (im_norm * 255).astype(np.uint8)

        # 单张图 eval，避免批量接口的版本差异
        masks, flows, styles, diams = model.eval(
            im_u8,
            diameter=float(cellpose_diameter),
            channels=list(cellpose_channel) if isinstance(cellpose_channel, (list,tuple)) else [0,0],
            flow_threshold=float(flow_threshold),
            cellprob_threshold=float(cellprob_threshold)
        )
        masks_all.append(masks)

        # 简短的可视化/调试统计（可注释掉）
        # print(f"[{idx}] masks={int(masks.max()) if masks is not None else 0}, "
        #       f"pred_diam={diams:.2f if isinstance(diams,(float,int)) else diams}")

    return masks_all

#denoise images
def denoise_segment_images(dapi_all_z_images, diameter=6/0.16, channels=[0,0], model_type="cyto3", restore_type="deblur_cyto3", nframes=1, int_thresh = 0.9999): #0-255 intensity values
    dapi_all_z = dapi_all_z_images

    dapi_merged = []
    for i in range(len(dapi_all_z)):
        print("Merging top 8 z images for FOV " +str(i) + "...")
        intensity = []
        for j in range(len(dapi_all_z[i])):
            intensity.append(np.mean(dapi_all_z[i][j].ravel()))
        intensity = intensity - np.min(intensity)
        z_pos = []
        for k in range(len(intensity)):
            if intensity[k]>int_thresh*np.max(intensity):
                if len(z_pos) <nframes:
                    z_pos.append(k)
        dapi_merged.append(np.sum(dapi_all_z[i][z_pos], axis = 0))
    print("Denoising images and segmenting cells...")


    model = denoise.CellposeDenoiseModel(gpu=True, model_type=model_type, restore_type=restore_type)
    masks, flows, styles, dapi_merged_dn = model.eval(dapi_merged, diameter=diameter, channels=channels)
    
    return masks # , flows, styles, dapi_merged_dn

def expand_masks(masks, n):
    expanded_masks= []
    for i in range(len(masks)):
        expanded = expand_labels(masks[i], distance=n)
        expanded_masks.append(expanded)
    return expanded_masks

def find_cell_center(image, cell_value):
    """
    Finds the center position of a cell in an image.

    Args:
    - image: a numpy array where each different value corresponds to a different cell.
    - cell_value: the value of the cell to find the center position for.

    Returns:
    - A tuple with the x and y coordinates of the center position of the cell.
    """
    indices = np.where(image == cell_value)
    center_x = np.mean(indices[0])
    center_y = np.mean(indices[1])
    return center_x, center_y

def map_gene2cell(genes_result_global, expanded_masks, position_reg, dim, xdim, ydim):
    genes_result_global['FOV'] = genes_result_global['FOV'].astype(int)
    fovs = np.unique(genes_result_global['FOV'])
    genes_result_cell = []

    # Label connected components in the mask image outside the loop
    labeled_masks = [sk.measure.label(mask) for mask in expanded_masks]

    for fov in range(len(fovs)):
        print("Mapping FOV " + str(fov))
        mask = expanded_masks[fov]
        cells = np.unique(mask)
        genes_result_fov = genes_result_global[genes_result_global['FOV'] == fov]
        genes_result_fov["cell_ID"] = mask[list(genes_result_fov['m1']), list(genes_result_fov['m2'])]
        genes_result_fov = genes_result_fov[genes_result_fov["cell_ID"] != 0]
        # Get properties of labeled regions for the current FOV
        props = sk.measure.regionprops(labeled_masks[fov])
        centroids = np.array([prop.centroid for prop in props])
        if len(centroids) == 0:
            continue
        # Assign center coordinates to genes_result_fov using cell numbers
        cell_num = genes_result_fov['cell_ID'].values - 1
        x_centers = np.array(centroids[cell_num.astype(int), 0])
        y_centers = np.array(centroids[cell_num.astype(int), 1])
        
        genes_result_fov["cell_center_X"] = x_centers
        genes_result_fov["cell_center_Y"] = y_centers
        genes_result_fov["cell_center_globX"] = x_centers + position_reg['X'][fov] - 2304/2
        genes_result_fov["cell_center_globY"] = y_centers - position_reg['Y'][fov] + np.max(position_reg['Y'])
        genes_result_cell.append(genes_result_fov)

    genes_result_cell = pd.concat(genes_result_cell)
    print("Creating Cell IDs...")
    
    genes_result_cell['cell_number'] = genes_result_cell['cell_ID'].astype(str) + '_' + genes_result_cell['FOV'].astype(str)
    genes_result_cell['cell_globID'] = genes_result_cell.groupby(['FOV', 'cell_ID']).ngroup()
    
    return genes_result_cell

# find neighbor fovs (only with larger index)
def find_neighb_fovs(m, position_reg, dimensions):
    neighbor = []
    for n in range(len(position_reg)):
        if (abs(position_reg.loc[n, 'X']-position_reg.loc[m, 'X']) < dimensions) & (abs(position_reg.loc[n, 'Y']-position_reg.loc[m, 'Y']) < dimensions) & (n > m): 
            y_m, x_m = position_reg.loc[m, 'X'], position_reg.loc[m, 'Y']
            y_n, x_n = position_reg.loc[n, 'X'], position_reg.loc[n, 'Y']
            if (x_m-x_n > dimensions*0.1) & (y_m-y_n > dimensions*0.1):
                neighbor.append((n, "top_right", "bottom_left"))
            elif (x_m-x_n > dimensions*0.1) & (abs(y_n-y_m) < dimensions*0.1):
                neighbor.append((n, "right", "left"))
            elif (x_m-x_n > dimensions*0.1) & (y_n-y_m > dimensions*0.1):
                neighbor.append((n, "bottom_right", "top_left"))
            elif (abs(x_n-x_m) < dimensions*0.1) & (y_n-y_m > dimensions*0.1):
                neighbor.append((n, "bottom", "top"))
            elif (x_n-x_m > dimensions*0.1) & (y_n-y_m > dimensions*0.1):
                neighbor.append((n, "bottom_left", "top_right"))
            elif (x_n-x_m > dimensions*0.1) & (abs(y_m-y_n) < dimensions*0.1):
                neighbor.append((n, "left", "right"))
            elif (x_n-x_m > dimensions*0.1) & (y_m-y_n > dimensions*0.1):
                neighbor.append((n, "top_left", "bottom_right"))
            elif (abs(x_m-x_n) < dimensions*0.1) & (y_m-y_n > dimensions*0.1):
                neighbor.append((n, "top", "bottom"))
    return neighbor

# find cell in the boundaries with genecalls
def find_cell_boundary(fov, mask, cells_mapped, boundaryType, dimensions, overlap=0.15):
    boundary = int(dimensions * overlap)
    xTemp, yTemp = np.where(mask != 0)
    indexTemp = list(zip(xTemp, yTemp))
    cells_boundary = []
    for (x, y) in indexTemp:
        if boundaryType == 'top_right':
            if (x<=boundary) & (y>=(dimensions-boundary)):
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
        elif boundaryType == 'right':
            if y>=(dimensions-boundary):
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
        elif boundaryType == 'bottom_right':
            if (x>=(dimensions-boundary)) & (y>=(dimensions-boundary)):
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
        elif boundaryType == 'bottom':
            if x>=(dimensions-boundary):
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
        elif boundaryType == 'bottom_left':
            if (x>=(dimensions-boundary)) & (y<=boundary):
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
        elif boundaryType == 'left':
            if y<=boundary:
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
        elif boundaryType == 'top_left':
            if (x<=boundary) & (y<=boundary):
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
        elif boundaryType == 'top':
            if x<=boundary:
                cells_boundary.append(str(mask[x, y])+'_'+str(fov))
    cells_boundary = np.unique(cells_boundary)
    cells_ret = list(set(cells_boundary).intersection(set(cells_mapped)))
    cells_result = [int(i.split('_')[0]) for i in cells_ret]
    return sorted(cells_result)

# check overlapped cells between two fovs
def overlap_condition_check(m, n, maskM, maskN, cellM, cellN, position_reg, dimensions):
    xpos_corr, ypos_corr = [int(i) for i in position_reg['X']], [int(i) for i in position_reg['Y']]
    width = np.max(xpos_corr)-np.min(xpos_corr)+int(dimensions)
    height = np.max(ypos_corr)-np.min(ypos_corr)+int(dimensions)
    half = int(dimensions/2)
    
    xindexM, yindexM = np.where(maskM == cellM) # np array
    xposM = xpos_corr[m] - half + xindexM
    yposM = height - ypos_corr[m] - half + yindexM
    posM = list(zip(xposM, yposM))
    xindexN, yindexN = np.where(maskN == cellN)
    xposN = xpos_corr[n] - half + xindexN
    yposN = height - ypos_corr[n] - half + yindexN
    posN = list(zip(xposN, yposN))
    ret = list(set(posM).intersection(set(posN)))

    # start conditions
    if len(posM) == 0 or len(posN) == 0:
        return False
    else:
        ratio = len(posM)/len(posN)
        if ratio > 0.5 and ratio < 2: 
            if len(ret)/min(len(posM), len(posN)) >= 0.5:# (len(ret)/len(posM) > 0.5) & (len(ret)/len(posN) > 0.5)
                return True
            else:
                return False
        else:
            if len(ret)/min(len(posM), len(posN)) >= 0.5:
                return True
            else:
                return False

# delete the cell with less genes
def overlap_condition_remove(m, n, cellM, cellN, gene_trimmed):
    gene_cellM = np.where(gene_trimmed['cell_number']==str(cellM)+'_'+str(m))[0]
    gene_cellN = np.where(gene_trimmed['cell_number']==str(cellN)+'_'+str(n))[0]
    if len(gene_cellM) > len(gene_cellN): # delete N
        # maskN[np.where(maskN==cellN)] = 0
        cell_num = str(cellN)+'_'+str(n)
    else:                                 # delete M
        # maskM[np.where(maskM==cellM)] = 0
        cell_num = str(cellM)+'_'+str(m)
    return cell_num

def overlap_rmv_cells_and_calls(gene_mapped, expanded_masks, position_reg, dimensions):
    fov_num = len(expanded_masks)
    cell_deleted = []
    for m in range(fov_num):
        neighb_fovs = find_neighb_fovs(m, position_reg, dimensions)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"main fov "+str(m)+":", neighb_fovs)
        for (n, bTypeM, bTypeN) in neighb_fovs:
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"    check neighbor fov "+str(n))
            maskM, maskN = expanded_masks[m], expanded_masks[n]
            cells_mapped = np.unique(gene_mapped['cell_number'])
            cells_resultM = find_cell_boundary(m, maskM, cells_mapped, bTypeM, dimensions)
            cells_resultN = find_cell_boundary(n, maskN, cells_mapped, bTypeN, dimensions)
            for cellN in cells_resultN:
                for cellM in cells_resultM:
                    if overlap_condition_check(m, n, maskM, maskN, cellM, cellN, position_reg, dimensions):
                        # print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"        repeated cells: ("+str(cellM)+"_"+str(m)+", "+str(cellN)+"_"+str(n)+") delete: "+cell_num)
                        cell_deleted.append(overlap_condition_remove(m, n, cellM, cellN, gene_mapped))
    trimmed_masks = copy.deepcopy(expanded_masks)
    for i in cell_deleted:
        cell_number, fov = int(i.split('_')[0]), int(i.split('_')[1])
        trimmed_masks[fov][np.where(trimmed_masks[fov]==cell_number)] = 0
    gene_trimmed = gene_mapped[~gene_mapped['cell_number'].isin(cell_deleted)]
    return gene_trimmed, trimmed_masks, cell_deleted

def overlap_rmv_subset(q_cell_deleted, fov, gene_mapped, mask_mapped, position_reg, dimensions):
    cell_deleted = []
    m = fov
    neighb_fovs = find_neighb_fovs(m, position_reg, dimensions)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"main fov "+str(m)+":", neighb_fovs)
    for (n, bTypeM, bTypeN) in neighb_fovs:
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"    check neighbor fov "+str(n))
        maskM, maskN = mask_mapped[m], mask_mapped[n]
        cells_mapped = np.unique(gene_mapped['cell_number'])
        cells_resultM = find_cell_boundary(m, maskM, cells_mapped, bTypeM, dimensions)
        cells_resultN = find_cell_boundary(n, maskN, cells_mapped, bTypeN, dimensions)
        for cellN in cells_resultN:
            for cellM in cells_resultM:
                if overlap_condition_check(m, n, maskM, maskN, cellM, cellN, position_reg, dimensions):
                    # print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"        repeated cells: ("+str(cellM)+"_"+str(m)+", "+str(cellN)+"_"+str(n)+") delete: "+cell_num)
                    cell_deleted.append(overlap_condition_remove(m, n, cellM, cellN, gene_mapped))
    q_cell_deleted.put(cell_deleted)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"    finish FOV: "+str(fov)+",", str(q_cell_deleted.qsize())+"/"+str(len(position_reg))+' fovs done')

def overlap_rmv_mp(gene_mapped, mask_mapped, position_reg, dim):
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"start multiprocessing; "+"number of cpu: "+str(mp.cpu_count()))
    p = Pool(processes=mp.cpu_count()) # cannot write Queue()
    q_cell_deleted = mp.Manager().Queue()
    for fov in range(len(mask_mapped)):
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"start multi process FOV: "+str(fov))
        # temp_gene = gene_mapped[gene_mapped['FOV']==fov]
        # temp_mask = mask_mapped[fov]
        p.apply_async(overlap_rmv_subset, args=(q_cell_deleted, fov, gene_mapped, mask_mapped, position_reg, dim))
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"Wait for all subprocesses done ...")
    p.close()
    p.join()
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"All subprocesses done")

    cell_deleted = []
    for i in range(q_cell_deleted.qsize()):
        cell_deleted.extend(q_cell_deleted.get())
    mask_trimmed = copy.deepcopy(mask_mapped)
    for i in cell_deleted:
        cell_number, fov = int(i.split('_')[0]), int(i.split('_')[1])
        mask_trimmed[fov][np.where(mask_trimmed[fov]==cell_number)] = 0
    gene_trimmed = gene_mapped[~gene_mapped['cell_number'].isin(cell_deleted)]
    return gene_trimmed, mask_trimmed, cell_deleted

def gene_expression_matrix(masks, spots, dim):
    spots_data = pd.DataFrame(columns = ['Cell number', 'Spot number', 'Spot X Pos', 'Spot Y Pos', 'Gene'])
    cells_data = pd.DataFrame(columns = ['Cell number', 'Cell Max X', 'Cell Max Y', 'Cell Min X', 'Cell Min Y'])
    
    num_cells = np.unique(masks)
    
    spots_cells = []
    spots_temp = []
    all_spots = []
    for f in range(len(np.unique(spots['FOV']))):
        num_cells = np.unique(masks[f])
        num_cells = [num_cells != 0]

        for i in range(len(num_cells)):
            masks_sc = np.where(masks[f] == i)
            cell_position = list(zip(masks_sc[0],masks_sc[1]))
            spots_temp = []
            
            points = list(zip((spots[spots['FOV']==f]['m2']),(dim-spots[spots['FOV']==f]['m1'])))
            spots_cells = list(set(points) & set(cell_position))
            spots_cells_ind = np.where(np.array(spots_cells) == np.array(points))[0]
            if len(spots_cells_ind) != 0:
                spots_temp = pd.DataFrame()
                cells_temp = pd.DataFrame()
                spots_temp['Cell number'] = [f+i]*len(spots_cells_ind)
                spots_temp['FOV'] = [f]*len(spots_cells_ind)
                cells_temp['Cell number'] =  [f+i]*len(spots_cells_ind)

                for m in spots_cells_ind:
                    spots_temp['Spot X Pos'] = spots[spots['FOV']==f].iloc[m]['m2']
                    spots_temp['Spot Y Pos'] = dim-spots[spots['FOV']==f].iloc[m]['m1']
                    spots_temp['Gene'] = spots[spots['FOV']==f].iloc[m]['Names']
                cells_temp['Cell Max X'] =  np.max(masks_sc[0])
                cells_temp['Cell Max Y'] =  np.max(masks_sc[1])
                cells_temp['Cell Min Y'] =  np.min(masks_sc[1])
                cells_temp['Cell Min X'] =  np.min(masks_sc[0])
                spots_data = pd.concat([spots_data, spots_temp])
                cells_data = pd.concat([cells_data, cells_temp])
    return spots_data, cells_data

def convert_to_annData(gene_mat_df):
    genes = np.unique(gene_mat_df['Names'])
    cells = np.unique(gene_mat_df['cell_number'])

    gene_counts = []
    coordinates = []
    for i in range(len(cells)):
        temp = gene_mat_df[gene_mat_df['cell_number']==cells[i]]
        gene_counts.append([len(np.array(temp[temp['Names']==gene])) for gene in genes])
        coordinates.append([np.array(temp['cell_center_globX'])[0], np.array(temp['cell_center_globY'])[0]])
    gene_expression_matrix = pd.DataFrame(gene_counts, columns = genes, index = cells)
    adata = ad.AnnData(gene_expression_matrix)
    adata.obs_names = cells
    adata.var_names = genes
    adata.obsm['spatial'] = np.array(coordinates)
    #adata.uns['spatial'] = np.array(coordinates)
    return adata

# def remove_overlap(gene_mat):
#     # List to store indices of points to drop
#     points_to_drop = set()
#     for FOV in np.unique(gene_mat['FOV']):
#         # Build a KDTree
#         xpoints = np.array(gene_mat[gene_mat['FOV']==FOV]['center_x'])
#         ypoints = np.array(gene_mat[gene_mat['FOV']==FOV]['center_y'])
#         points = np.stack([xpoints, ypoints], axis = 1)
#         # Iterate through each point to find nearby points
#         for i,point in enumerate(points):
#             if ypoints[i] > 2000 or ypoints[i] < 200 or xpoints[i] > 2000 or xpoints[i] <200:
#                 cell_number = gene_mat[gene_mat['center_y'] == ypoints[i]]['cell_number'].values[0]
#                 # remove all points in fov from 'points'
#                 fov = int(cell_number.split('_')[1])
#                 xpoints_fov = np.array(gene_mat[gene_mat['FOV'] != fov]['center_x'])
#                 ypoints_fov = np.array(gene_mat[gene_mat['FOV'] != fov]['center_y'])
#                 points_otherfovs = np.stack([xpoints_fov, ypoints_fov], axis = 1)
#                 kdtree = KDTree(points_otherfovs)
#                 nearby_indices = kdtree.query(point, k=1)
#                 # if number of cells near this cell is greater than 1
#                 j = 0
#                 if nearby_indices[0] < 70:
#                     #find all genes in nearby cells
#                     genes_cell = gene_mat[gene_mat['center_y'] == ypoints[i]]
#                     genes_nearbycell = gene_mat[gene_mat['center_y'] == points_otherfovs[nearby_indices[1]][1]]
#                     xgenes = genes_nearbycell['globX']
#                     ygenes = genes_nearbycell['globY']
#                     gene_points = np.stack([xgenes, ygenes], axis =1)
#                     kdtree = KDTree(gene_points)
#                     # potential issues: if shift of stitching is too big, if dots are too dense?                   
#                     # create a tree and evaluate if closest nearby neighbor is gene of interest
#                     for k in range(len(genes_cell)):
#                         poi = [genes_cell.iloc[k]['globX'], genes_cell.iloc[k]['globY']]
#                         dist, nearby_gene_indices = kdtree.query(poi, k=1)
#                         if genes_nearbycell.iloc[nearby_gene_indices]['Names'] == genes_cell.iloc[k]['Names']:
#                             j = j+1
#                     if j>3:
#                         drop_indices = gene_mat[gene_mat['center_y'] == points_otherfovs[nearby_indices[1]][1]].index.values 
#                         points_to_drop.update(drop_indices)

#         # Drop duplicate points based on indices
#     genes_result_rmvOverlap = gene_mat.drop(index=points_to_drop).reset_index(drop=True)

#     return genes_result_rmvOverlap, points_to_drop