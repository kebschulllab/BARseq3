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

def create_codebook_num(filepath_codebook, round_num_for_gene_call, base_code): # working with round num
    genenames, gene_codes = [], []
    with open(filepath_codebook, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].__contains__(codecs.BOM_UTF8.decode(f.encoding)):
            # A Byte Order Mark is present
                genenames.append(row[0].strip(codecs.BOM_UTF8.decode(f.encoding)))
            else:
                genenames.append(row[0])
            gene_codes.append(row[1][0:round_num_for_gene_call])
    check_head = set(gene_codes[0])
    available = set('ATGC')
    if not check_head.issubset(available):
        genenames.pop(0)
        gene_codes.pop(0)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"genenames: ", genenames)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"gene_codes: ", gene_codes)

    codebook = np.full((round_num_for_gene_call, 4, len(gene_codes)), False)
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

def create_codebook(filepath_codebook, round_num_for_gene_call, base_code): # working with round index
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
                barcode_subset = barcode[round_num_for_gene_call]
                # print(barcode_subset)
                gene_codes.append(barcode_subset)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"genenames: ", genenames)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"gene_codes: ", gene_codes)

    codebook = np.full((len(round_num_for_gene_call), 4, len(gene_codes)), False)
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



def find_params_gradient(rounds_all, codeflat, genenames, nogene_keyword, filepath_val, fov_minmax, round_num_for_gene_call, fdr_thresh):
    # rounds_all = image_preped
    R = rounds_all[0].shape[0]
    C = rounds_all[0].shape[1]
    rounds_all = np.stack(rounds_all, axis = 0)
    rounds_all = rounds_all[fov_minmax,:]
    #rounds_all = rounds_all[:,round_num_for_gene_call]
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
    # thresh = 0.6
    if np.isnan(thresh):
        thresh = 0.7
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+'    initial thresh from calculation: '+str(thresh))

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+'refine threshold by gradient convergence')
    diff = 2.0
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


def find_peaks_in_image(image, codeflat, thresh_refined, noisefloor, len_wid, FOV):
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
         et=bardensr.spot_calling.estimate_density_singleshot(
                 Xflat,
                 codeflat,
                 noisefloor
         )
         #et=bardensr.spot_calling.estimate_density_iterative(Xflat,codebook,l1_penalty=0,psf_radius=(np.int64(0),np.int64(0),np.int64(0)),
         #                                                    iterations=100,estimate_codebook_gain=True,
         #                                                    rounds=None,estimate_colormixing=False, estimate_phasing=False,
         #                                                    use_tqdm_notebook=False)[0]

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

def call_genes_large_data(image, positions_glob, codeflat, codebook, genenames,thresh_refined, noisefloor, len_wid, round_num_for_gene_call):
     all_spots_fov = []
     image = np.stack(image, axis = 0)
     all_spots_fov_df = pd.DataFrame()
     #image = image[:,round_num_for_gene_call]
     for i in range(len(image)):
         genes_result, positions = find_peaks_in_image(image[i], codeflat, thresh_refined, noisefloor, len_wid, i)
         all_spots_df = pd.DataFrame()
         #genes_result['globX'] = 0
         #genes_result['globY'] = 0
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

