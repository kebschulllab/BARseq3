"""
load genebook, find threshold for genecalling, call genes
"""
import bardensr
import numpy as np
import pandas as pd
import csv
import codecs
import matplotlib.pyplot as plt
from datetime import datetime
# print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"")

# import copy
# from skimage import io
# from skimage.color import rgb2gray
# from skimage.data import stereo_motorcycle, vortex
# from skimage.transform import warp
# from skimage.registration import optical_flow_tvl1, optical_flow_ilk
# import IPython.display
# import SimpleITK as sitk
# import sitkibex
# import tensorflow

def create_codebook_num(filepath_codebook, round_num, base_code): # working with round num
    genenames, gene_codes = [], []
    with open(filepath_codebook, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].__contains__(codecs.BOM_UTF8.decode(f.encoding)):
            # A Byte Order Mark is present
                genenames.append(row[0].strip(codecs.BOM_UTF8.decode(f.encoding)))
            else:
                genenames.append(row[0])
            gene_codes.append(row[1][0:round_num])
    check_head = set(gene_codes[0])
    available = set('ATGC')
    if not check_head.issubset(available):
        genenames.pop(0)
        gene_codes.pop(0)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"genenames: ", genenames)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"gene_codes: ", gene_codes)

    codebook = np.full((round_num, 4, len(gene_codes)), False)
    for tagNum, tag in enumerate(gene_codes):
        for roundNum, letter in enumerate(tag):
            if letter == base_code[0]: 
                codebook[roundNum, 0, tagNum] = True
            if letter == base_code[1]: 
                codebook[roundNum, 1, tagNum] = True
            if letter == base_code[2]: 
                codebook[roundNum, 2, tagNum] = True
            if letter == base_code[3]: 
                codebook[roundNum, 3, tagNum] = True
    codeflat = codebook.reshape((codebook.shape[0]*codebook.shape[1],-1)) # R,C,J=codebook.shape
    
    return codebook, genenames, codeflat

def create_codebook(filepath_codebook, round_num, base_code): # working with round index
    genenames, gene_codes = [], []
    with open(filepath_codebook, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            check_head = set(row[1])
            available = set('ATGC')
            if check_head.issubset(available):
                if row[0].__contains__(codecs.BOM_UTF8.decode(f.encoding)):
                # A Byte Order Mark is present
                    genenames.append(row[0].strip(codecs.BOM_UTF8.decode(f.encoding)))
                else:
                    genenames.append(row[0])
                barcode = np.array([*row[1]])
                # print(barcode)
                barcode_subset = barcode[round_num]
                # print(barcode_subset)
                gene_codes.append(barcode_subset)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"genenames: ", genenames)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"gene_codes: ", gene_codes)

    codebook = np.full((len(round_num), 4, len(gene_codes)), False)
    for tagNum, tag in enumerate(gene_codes):
        for roundNum, letter in enumerate(tag):
            if letter == base_code[0]: 
                codebook[roundNum, 0, tagNum] = True
            if letter == base_code[1]: 
                codebook[roundNum, 1, tagNum] = True
            if letter == base_code[2]: 
                codebook[roundNum, 2, tagNum] = True
            if letter == base_code[3]: 
                codebook[roundNum, 3, tagNum] = True
    codeflat = codebook.reshape((codebook.shape[0]*codebook.shape[1],-1)) # R,C,J=codebook.shape
    return codebook, genenames, codeflat



def find_params_gradient(rounds_all, codeflat, genenames, nogene_keyword, filepath_val, fov_minmax, round_num, fdr_thresh):
    # rounds_all = image_preped
    R = rounds_all[0].shape[0]
    C = rounds_all[0].shape[1]
    rounds_all = np.stack(rounds_all, axis = 0)
    rounds_all = rounds_all[fov_minmax,:]
    #rounds_all = rounds_all[:,round_num]
    noisefloor = 0.05
    ercc_names = [s for s in genenames if nogene_keyword in s]
    ercc_codes = np.where(np.in1d(genenames, ercc_names))[0]
    print("False Detection Rate: "+str(fdr_thresh))
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+'initialize thresh level for optimization')
    ##\thresh = 0.7
    ercc_maxes = []
    for i in range(len(rounds_all)):
        Xflat = rounds_all[i].reshape(R*C,1,rounds_all[i].shape[-2],rounds_all[i].shape[-1])
        #Xflat = np.expand_dims(Xflat, axis=1)
        print(codeflat.shape)
        print(Xflat.shape)
        et = bardensr.spot_calling.estimate_density_singleshot(Xflat[:,:,500:1000, 500:1000], codeflat, 0.01)  # <-- NOTE: we could save this et because we'll use it later
                                                                                        # but computing et isn't actually much slower than loading it from disk... :)
        ercc_maxes.append(et[:,:,:,ercc_codes].max(axis=(0,1,2))) # max value (single value for each no gene) of the evidence tensor for each fov
    ercc_maxes = np.array(ercc_maxes) # This is the max values of all unused barcodes in all fovs. 
    thresh = np.median(np.median(ercc_maxes, axis=1)) # median across fovs of the median across no genes # this is the initial thresh level for optimization
    if np.isnan(thresh):
        thresh = 0.7
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+'    initial thresh from calculation: '+str(thresh))

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+'refine threshold by gradient convergence')
    diff = 1.0
    stepsize = 0.1
    thresh_list, fdr_list =[], []
    while abs(diff) > 0.01: # diff = 5% - threshold
        nogenes_num, genes_num = [], []
        nogenes_calls, genes_calls = [], []
        for i in range(len(rounds_all)):
            #Xflat = rounds_all[i].reshape((R*C,)+rounds_all[i].shape[-2:])
            #Xflat = np.expand_dims(Xflat, axis=1)
            # print('before')
            #et = bardensr.spot_calling.estimate_density_singleshot(Xflat, codeflat, noisefloor)
            #spots = bardensr.spot_calling.find_peaks(et, thresh) # , use_tqdm_notebook=True
            # print('after')
            spots, positions = find_peaks_in_image(rounds_all[i], codeflat, thresh, noisefloor, 500, i)
            spots = pd.concat(spots)
            no_genes_called = len(np.unique(spots.iloc[np.where(np.isin(spots['j'],ercc_codes))]['j'])) + 0.01
            nogenes_num.append(no_genes_called)
            genes_called = len(np.unique(spots['j'])) - no_genes_called + 0.01
            genes_num.append(genes_called)
            ercc_c = len(spots.iloc[np.where(np.isin(spots['j'],ercc_codes))]['j'])
            nogenes_calls.append(ercc_c)
            genes_calls.append(len(spots)-ercc_c)
        # divide by all gene num
        nogene_avg = np.mean(nogenes_calls)/len(ercc_names)
        trgene_avg = np.mean(genes_calls)/(len(genenames)-len(ercc_names))
        fdrmean = nogene_avg / (trgene_avg + 0.001)
        # # divide by unique gene num
        # fdrmean = (np.mean(nogenes_calls)/np.mean(nogenes_num))*(np.mean(genes_num)/np.mean(genes_calls))
        thresh_list.append(thresh)
        fdr_list.append(fdrmean)
        diff = fdr_thresh - fdrmean
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+'    thresh: '+str(thresh)+'; fdrmean: '+str(fdrmean)+'; diff:'+str(diff))
        thresh -= stepsize * diff / abs(diff)
        stepsize /= 2
    thresh_refined = thresh_list[-1]
    with open(filepath_val+'thresh_refined.txt','w') as f:
        f.write(str(thresh_refined))
    plt.figure(figsize=(10,10))
    plt.scatter(thresh_list, fdr_list)
    plt.plot(thresh_list, fdr_list)
    plt.savefig(filepath_val+'find_params_threshold_gradient.png')

    return thresh_refined, noisefloor


def call_genes(threshold, noisefloor, rounds_all, codeflat, genenames, positions_reg, poolsize, xdim, ydim, dim):
    R = rounds_all[0].shape[0]
    C = rounds_all[0].shape[1]
    genenames = np.array(genenames)
    genes_result = pd.DataFrame()
    for i in range(len(rounds_all)):
        Xflat = rounds_all[i].reshape((R*C,)+rounds_all[i].shape[-2:])
        Xflat = np.expand_dims(Xflat,axis=1)
        et = bardensr.spot_calling.estimate_density_singleshot(Xflat, codeflat, noisefloor)
        genes_result_sing = bardensr.spot_calling.find_peaks(et,threshold, poolsize)# , use_tqdm_notebook=True
        genes_result_sing['FOV'] = [i]*len(genes_result_sing['m2'])
        genes_result_sing['Names'] = genenames[genes_result_sing['j']]
        # temp['globX'] =  np.array(positions_corr['X'])[i] + temp['m2']
        # temp['globY'] =  np.array(positions_corr['Y']).max() - np.array(positions_corr['Y'])[i] + dim - temp['m1']
        genes_result_sing['globX'] = ((genes_result_sing['m1'])*(dim/xdim))+positions_reg['X'][i]-dim/2
        genes_result_sing['globX'] = [int(j) for j in genes_result_sing['globX']]
        genes_result_sing['globY'] = np.max(positions_reg['Y'])-positions_reg['Y'][i]+((genes_result_sing['m2'])*(dim/ydim))
        genes_result_sing['globY'] = [int(j) for j in genes_result_sing['globY']]
        genes_result_sing.drop(['m0', 'j'], axis=1, inplace=True)
        genes_result = pd.concat([genes_result, genes_result_sing])
    return genes_result


def extract_patches(image, patch_size, overlap):
     # calculate the number of patches that can be extracted
     n_patches_x = int(np.ceil((image.shape[3]-patch_size[3])/(patch_size[3]*(1-overlap))))+1
     n_patches_y = int(np.ceil((image.shape[4]-patch_size[4])/(patch_size[4]*(1-overlap))))+1
     # initialize empty list to store patches
     patches = []
     for i in range(n_patches_x):
         for j in range(n_patches_y):
             x = int(i*patch_size[3]*(1-overlap))
             y = int(j*patch_size[4]*(1-overlap))
             patch = image[:,:,:,x:x+patch_size[3],y:y+patch_size[4]]
             patches.append(patch)
     # return list of patches
     return patches


def find_peaks_in_image(image, codeflat, codebook, thresh_refined, noisefloor, len_wid, FOV):
     patch_size = [1, image.shape[1], image.shape[2], len_wid, len_wid]
     overlap = 0.1
     all_spots = []
     image = np.expand_dims(image, axis=0)
     patches = extract_patches(image, patch_size, overlap)
     n_patches_x = int(np.ceil((image.shape[3]-patch_size[3])/(patch_size[3]*(1-overlap))))+1

     n_patches_y = int(np.ceil((image.shape[4]-patch_size[4])/(patch_size[4]*(1-overlap))))+1
     # Find peaks and positions in each patch
     peaks = []
     positions = []
     for i in range(len(patches)):
         patch = patches[i]
         print("Patch "+str(i)+" out of "+ str(len(patches)) + " for FOV "+str(FOV))
         Xflat = patch.reshape(patch.shape[1]*patch.shape[2],1,patch.shape[3], patch.shape[4])
         #patch = tf.convert_to_tensor(patch)
        # '''et=bardensr.spot_calling.estimate_density_singleshot(
         #        Xflat,
        #         codeflat,
         #        noisefloor
         #)'''
         et=bardensr.spot_calling.estimate_density_iterative(Xflat,codeflat,l1_penalty=0,psf_radius=(np.int64(0),np.int64(0),np.int64(0)),
                                                             iterations=100,estimate_codebook_gain=True,
                                                             rounds=None,estimate_colormixing=False, estimate_phasing=False,
                                                             use_tqdm_notebook=False)[0]

         spots=bardensr.spot_calling.find_peaks(et,thresh_refined,poolsize=(0,3,3), use_tqdm_notebook=False)
         all_spots.append(spots)
         #positions.append([i // (image.shape[3]//patch_size[3]), i % (image.shape[4]//patch_size[4])])

     # Stitch the patches back together
     patch_count = 0
     for i in range(n_patches_x):
         for j in range(n_patches_y):
             x = int(i*patch_size[3]*(1-overlap))
             y = int(j*patch_size[4]*(1-overlap))
             positions.append([x,y])
     return all_spots, positions

#Chris
def call_genes_tiled_with_overlap(
     Xflat, codeflat, tile_size=1024, overlap=64, iterations=10, **kwargs
 ):
     """
     Run estimate_density_iterative_fp32 in spatial tiles with optional overlap padding.
     Returns a full (H, W, J) tensor.
     """

 
     print(Xflat.shape)
     RC,_, H, W = Xflat.shape
     print("Image shape:", H, W)
     print("Codeflatshape:", codeflat.shape)
     import numpy as np

     # Assuming codeflat is initially 12x5
     #codeflat = np.roll(codeflat, shift=1, axis=0)  # Roll elements along axis 0 (rows)
     #codeflat = codeflat.reshape(5, 12)  # Now reshape to 5x12
     #codeflat = codeflat.astype(np.float32)  # Ensure it's float32
     J = codeflat.shape[-1]
     print("Codeflatshape:", codeflat.shape)

     result_full = np.zeros((H, W, J), dtype=np.float32)
 
     for i in range(0, H, tile_size):
         for j in range(0, W, tile_size):
             # Compute padded tile bounds
             i_start = max(i - overlap, 0)
             i_end = min(i + tile_size + overlap, H)
             j_start = max(j - overlap, 0)
             j_end = min(j + tile_size + overlap, W)
 
             # Extract padded tile from all frames
             Xtile = Xflat[:, :, i_start:i_end, j_start:j_end]
             print(f"Processing tile ({i}:{i_end}, {j}:{j_end}), shape: {Xtile.shape}")
             try:
                 et_tile, _ = bardensr.spot_calling.estimate_density_iterative(
                     Xtile, codeflat, iterations=iterations, **kwargs
                 )
                 print("et_tile shape:", et_tile.shape)
                 et_tile = et_tile.squeeze(axis=0) # Remove batch dimension
             except Exception as e:
                 print(f"Tile ({i}:{i_end}, {j}:{j_end}) failed: {e}")
                 continue
 
             # Determine where to crop the tile output
             crop_top = i - i_start
             crop_bottom = crop_top + min(tile_size, H - i)
             crop_left = j - j_start
             crop_right = crop_left + min(tile_size, W - j)
 
             et_cropped = et_tile[crop_top:crop_bottom, crop_left:crop_right, :]
 
             result_full[i:i+et_cropped.shape[0], j:j+et_cropped.shape[1], :] = et_cropped
 
     # return np.squeeze(result_full, axis=0) if result_full.shape[0] == 1 else result_full
     return result_full[np.newaxis, ...]# if result_full.shape[0] == 1 else result_full
 
ESTIMATE_DENSITY_ITERATIONS =10
import os

def call_genes_large_data_all(image, codeflat, codebook, genenames,thresh_refined, noisefloor, len_wid, round_num, fov_sample, find_thresh = False):
 
     os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
     #image = image[round_num]
     image = np.stack(image, axis = 0)
     # image = image.astype(np.float32, copy=False) 
     print(len(image))
     print(image[0].shape)
     Xflat = image.reshape(image.shape[0]*image.shape[1],1,image.shape[2], image.shape[3])
     print("Xflat shape:", Xflat.shape)
     # Xflat = image.reshape(image.shape[0]*image.shape[1],1,image.shape[2], image.shape[3]).astype(np.float32)
     # Xflat = Xflat.astype(np.float32)
     # codeflat = codeflat.astype(np.float32)
 
     print("Image dtype:", image.dtype)
     print("codeflat dtype:", codeflat.dtype)
     print("thresh_refined dtype:", np.asarray(thresh_refined).dtype)
 
 
     # et=bardensr.spot_calling.estimate_density_singleshot(Xflat,codeflat,0.05)
     # et,extra_learned_params=\
     # bardensr.spot_calling.estimate_density_iterative(Xflat,codeflat,use_tqdm_notebook=False,iterations=10)
     # et,extra_learned_params=estimate_density_iterative_fp32(Xflat,codeflat,use_tqdm_notebook=False,iterations=10)
     if ESTIMATE_DENSITY_ITERATIONS > 0:
         et = call_genes_tiled_with_overlap(Xflat, codeflat, tile_size=512, iterations=ESTIMATE_DENSITY_ITERATIONS)
     else:
         et = bardensr.spot_calling.estimate_density_singleshot(Xflat,codeflat,0.05)
     print("et shape:", et.shape)
     spots=bardensr.spot_calling.find_peaks(et,thresh_refined,poolsize=(1,3,3), use_tqdm_notebook=False)
 
 
     et_squeezed = et.squeeze(0)  
     # Extract the pixel intensity for each row in the DataFrame
     #pixel_values = spots.apply(lambda row: et_squeezed[int(row['m2']), int(row['m1']), int(row['j'])], axis=1)    
     # Add the pixel value as a new column
     #spots['pixel_val'] = pixel_values
     # spots['pixel_val'] = pd.Series(pixel_values.values, index=spots.index)
     
     # Compute the j index with the maximum value at each (m1, m2)
     max_j_array = np.argmax(et_squeezed, axis=2) # shape: (2304, 2304)
 
     # Add a column to DataFrame indicating if that row is the max j at that (m1, m2)
     print(spots)
     print(max_j_array.shape)
     spots['is_max_j'] = spots.apply(
         lambda row: row['j'] == max_j_array[int(row['m1']), int(row['m2'])],
         axis=1
     )
 
     # Keep only those rows
     spots = spots[spots['is_max_j']].drop(columns='is_max_j')
 
     print(spots)
     spots['FOV'] = [fov_sample]*len(spots['m2'])
     spots.drop_duplicates()
     print('Creating gene genenames column...')
     genenames = np.array(genenames)
     spots['Names'] = genenames[spots['j']]
     if find_thresh == False:
         spots.to_csv("Y:\Huihui\BARseq2\BARseq3_ana\Brain1\STARmap_output\gene_calling_test\gene_called_fov"+str(fov_sample)+".csv", index=False)
     if find_thresh == True:
         return spots
# original function
def call_genes_large_data(image, positions_glob, codeflat, codebook, genenames,thresh_refined, noisefloor, len_wid, round_num):
     all_spots_fov = []
     image = np.stack(image, axis = 0)
     print(image[0].shape)
     all_spots_fov_df = pd.DataFrame()
     ##image = image[:,round_num]
     for i in range(len(image)):
        genes_result, positions = find_peaks_in_image(image[i], codeflat, codebook, thresh_refined, noisefloor, len_wid, i)
        #spots = call_genes_large_data(image[i], codeflat, codebook, genenames,thresh_refined, noisefloor, len_wid, round_num, 0)
        all_spots_df = pd.DataFrame()
        print()
        all_spots_df = call_genes_large_data_all(image[i], codeflat, codebook, genenames,thresh_refined, noisefloor, len_wid, round_num, 0)

        genes_result['globX'] = 0
        genes_result['globY'] = 0
        print("Gene calling for FOV: "+str(i))
        for j in range(len(genes_result)):
              genes_result[j]['m1'] = genes_result[j]['m1'] + positions[j][0]
              genes_result[j]['m2'] = genes_result[j]['m2'] + positions[j][1]
              genes_result[j]['FOV'] = [i]*len(genes_result[j]['m2'])
              genes_result[j]['globX'] = genes_result[j]['m1'] + positions_glob['Y'][i]
              genes_result[j]['globX'] = [int(j) for j in genes_result[j]['globX']]
              genes_result[j]['globY'] = genes_result[j]['m2'] + np.max(positions_glob['X'])- positions_glob['X'][i]
              genes_result[j]['globY'] = [int(j) for j in genes_result[j]['globY']]
              all_spots_df = pd.concat([all_spots_df, genes_result[j]])
        all_spots_df.drop_duplicates()
        all_spots_fov.append(all_spots_df)
        for k in range(len(all_spots_fov)):
            all_spots_fov_df = pd.concat([all_spots_fov_df, all_spots_fov[k]])

     print('Creating gene genenames column...')
     genenames = np.array(genenames)
     all_spots_fov_df['Names'] = genenames[all_spots_fov_df['j']]
     all_spots_fov_df = all_spots_fov_df.drop_duplicates()
     return all_spots_fov_df

