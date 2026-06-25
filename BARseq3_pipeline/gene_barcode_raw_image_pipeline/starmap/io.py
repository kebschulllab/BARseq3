"""
input/output hdf5 file
read in round images, read in dapi images
load dapi images, load cell masks
plot gene calls over full dapi image
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pims_nd2 import ND2_Reader
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300 # set resolution
from datetime import datetime
import tifffile
from cellpose import utils
# import copy
# import cv2
# from PIL import Image

def save_hdf5(matrix, file_path): #assumes this is a list of np arrays
    f = h5py.File(file_path,'w')
    grp = f.create_group('list')
    for i,arr in enumerate(matrix):
        grp.create_dataset(str(i), data=arr)
    f.close()
    return None

def open_hdf5(file_path): #assumes this is a list of np arrays
    var_name= []
    f = h5py.File(file_path)
    data = f['list']
    for i in range(len(data)):
        var_name.append(data[str(i)][:])
    f.close()
    print(len(var_name))
    print(len(var_name[0]))
    return var_name

def open_hdf5_NxRxC(file_path, dim_n, dim_r): #assumes this is a list of np arrays
    print(dim_n)
    print(dim_r)
    #changed to allow reading in big files
    print("start loading:", file_path)
    final = [] # [N][R, 4, xdim, ydim])
    f = h5py.File(file_path)
    data = f['list']
    for i in range(dim_n):
        temp = []
        for j in range(dim_r):
            temp.append(data[str(i)][j])
        final.append(np.stack(temp,axis = 0))
        print(str(((i+1)*dim_r*100)/(dim_n*dim_r))+ "% done loading hdf5 file")
    # final = np.stack(final, axis =0)
    # final = np.squeeze(final)
    # print(final.shape)
    f.close()
    return final

def read_gtca_images(filename):
    print("start loading:", filename)
    image = ND2_Reader(filename)
    rounds = []
    # xpos, ypos = [], []
    with image as frames:
        frames.iter_axes = 'm'  # 't' is the default already
        frames.bundle_axes = 'cyx'  # when 'z' is available, this will be default
        # frames.default_coords['c'] = 0  # 0 is the default setting
        for frame in frames:
            # xpos.append(frame.metadata['y_um']/0.1612) #0.16 px/um
            # ypos.append(frame.metadata['x_um']/0.1612)
            rounds.append([frame[0], frame[1], frame[2], frame[3]])
    return rounds

def read_dapi_images(filename):
    print("start loading:", filename)
    if filename != "":
        image = ND2_Reader(filename)
        images, xpos, ypos = [], [], []
        # images_smaller = []
        with image as frames:
            frames.iter_axes = 'm'  # 't' is the default already
            frames.bundle_axes = 'yx' #FIGURE THIS OUT # when 'z' is available, this will be default
            frames.default_coords['c'] = 1  # 0 is the default setting
            for frame in frames:
                xpos.append(frame.metadata['y_um']/0.1612) #0.16 px/um
                ypos.append(frame.metadata['x_um']/0.1612)
                images.append(frame)
                # frame2 = cv2.resize(frame, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
                # images_smaller.append(frame2)
        positions = pd.DataFrame()
        positions['X'] = xpos
        positions['Y'] = ypos
        return [images, positions]
    else:
        return [None, None]
    
def read_dapi_all_z_images(filename):
    print("start loading:", filename)
    image = ND2_Reader(filename)
    images, xpos, ypos = [], [], []
    # images_smaller = []
    with image as frames:
        frames.iter_axes = 'm'  # 't' is the default already
        frames.bundle_axes = 'zxy' # # when 'z' is available, this will be default
        frames.default_coords['c'] = 1  # 0 is the default setting
        i=0
        for frame in frames:
            print("Reading FOV "+str(i))
            xpos.append(frame.metadata['y_um']/0.1612) #0.16 px/um
            ypos.append(frame.metadata['x_um']/0.1612)
            images.append(frame)
            i=i+1
            # frame2 = cv2.resize(frame, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
            # images_smaller.append(frame2)
    positions = pd.DataFrame()
    positions['X'] = xpos
    positions['Y'] = ypos
    return [images, positions]

def load_dapi(img_list, positions_corr, dimensions):
    xpos_corr, ypos_corr = [int(i) for i in positions_corr['X']], [int(i) for i in positions_corr['Y']]
    width = np.max(xpos_corr)-np.min(xpos_corr)+int(dimensions)
    height = np.max(ypos_corr)-np.min(ypos_corr)+int(dimensions)
    images = np.ones([width,height])
    half = int(dimensions/2)
    for i in range(len(img_list[:])):
        images[int(xpos_corr[i])-half:int(xpos_corr[i])+half, height-int(ypos_corr[i])-half:height-int(ypos_corr[i])+half] = img_list[i]
    images = np.clip(images, 0, 2000)
    return images

def load_mask(img_list, positions_corr, dimensions):
    xpos_corr, ypos_corr = [int(i) for i in positions_corr['X']], [int(i) for i in positions_corr['Y']]
    width = np.max(xpos_corr)-np.min(xpos_corr)+int(dimensions)
    height = np.max(ypos_corr)-np.min(ypos_corr)+int(dimensions)
    half = int(dimensions/2)
    images = np.zeros([width,height]) # there are cells with value 1
    for i in range(len(img_list)):
        # for x in range(dimensions):
        #     for y in range(dimensions):
        #         if images[xpos_corr[i]-half+x, height-ypos_corr[i]-half+y] == 0:
        #             images[xpos_corr[i]-half+x, height-ypos_corr[i]-half+y] = int(img_list[i][x, y])
        images[xpos_corr[i]-half:xpos_corr[i]+half, height-ypos_corr[i]-half:height-ypos_corr[i]+half] = images[xpos_corr[i]-half:xpos_corr[i]+half, height-ypos_corr[i]-half:height-ypos_corr[i]+half] + img_list[i]
    images = np.clip(images, 0, 2000)
    return images

def write_registration_file(data_rawnpy, round_num, fov_num, filepath_imageJ):
    # Write channel tiff files that are not aligned to 'round _' folders
    # i-round; j-fov; k-channel
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"write channel tiff files")
    for i in range(round_num):
        if not os.path.exists(filepath_imageJ+"round"+str(i+1)+"/"):
            os.makedirs(filepath_imageJ+"round"+str(i+1)+"/")
        if not os.path.exists(filepath_imageJ+"round"+str(i+1)+"/transf/"):
            os.makedirs(filepath_imageJ+"round"+str(i+1)+"/transf/")
        for j in range(fov_num):
            for k in range(4):
                tifffile.imwrite(filepath_imageJ+"round"+str(i+1)+'/data_chalign_r'+str(i+1)+'_fov'+str(j)+'_c'+str(k)+'.tiff', data_rawnpy[j][i,k])

def read_registration_file(shape, round_align, filepath_imageJ):
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"read ImageJ registration files")
    data_align = np.empty(shape)
    for FOV in range(shape[0]):
        for R in range(shape[1]):
            if R == round_align-1:
                data_align[FOV, R] = tifffile.imread(filepath_imageJ+"round"+str(R+1)+'/data_chalign_r'+str(R+1)+'_fov'+str(FOV)+'_chcorr.tiff')
            else:
                data_align[FOV, R] = tifffile.imread(filepath_imageJ+"round"+str(R+1)+'/data_chalign_r'+str(R+1)+'_fov'+str(FOV)+'_roundcorr.tiff')
    return data_align

def write_dapi_registration_file(data_rawnpy, fov_num, filepath_imageJ, subfolder):
    # Write channel tiff files that are not aligned to 'round _' folders
    # i-round; j-fov; k-channel
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"write tiff files for registration in "+subfolder)
    if not os.path.exists(filepath_imageJ + subfolder):
        os.makedirs(filepath_imageJ + subfolder)
    for j in range(fov_num):
        tifffile.imwrite(filepath_imageJ + subfolder + 'align'+ '_fov' + str(j) + '_raw.tiff', data_rawnpy[j])

def read_dapi_registration_file(shape, filepath_imageJ, subfolder): #Edited to include variable for change aligned rounds
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"read registration files from "+subfolder)
    nissl_align = np.empty(shape)
    for FOV in range(shape[0]):
           nissl_align[FOV] = tifffile.imread(filepath_imageJ+ subfolder + 'align_fov' + str(FOV) + '_aligned.tiff')
    return nissl_align

# def load_gene(gene_result, positions_corr, xdim, ydim, dimensions):
#     xpos_corr, ypos_corr = positions_corr['X'], positions_corr['Y']
#     spots = []
#     for i in np.unique(gene_result['FOV']):
#         spots.append(gene_result[gene_result['FOV']==i])
#     spots_list = copy.deepcopy(spots)
#     spots_concat = pd.DataFrame()
#     for i in range(len(spots)):
#         # spots_list[i]['m2'] = np.max(ypos_corr)-ypos_corr[i]+((spots_list[i]['m2'])*(dimensions/ydim))
#         # spots_list[i]['m2'] = [int(j) for j in spots_list[i]['m2']]
#         # spots_list[i]['m1'] = ((spots_list[i]['m1'])*(dimensions/xdim))+xpos_corr[i]-dimensions/2
#         # spots_list[i]['m1'] = [int(j) for j in spots_list[i]['m1']]
#         spots_concat = pd.concat([spots_concat, spots_list[i]])
#     return spots_concat

def plot_gene_dapi(gene_result, img_list, positions_corr, gene_names, xdim, ydim, dimensions, filepath_val, alpha=0.2, size=2):
    dapi_image = load_dapi(img_list, positions_corr, dimensions)
    # gene_image = load_gene(gene_result, positions_corr, xdim, ydim, dimensions)
    plt.figure(figsize=(30,30))
    plt.imshow(dapi_image, vmin = 0, vmax = 1000,alpha=alpha)
    for i in range(len(gene_names)):
        gene_pos = gene_result[gene_result['Names'] == gene_names[i]]
        plt.scatter(gene_pos['globY'], gene_pos['globX'], s=size, label=gene_names[i], cmap='viridis')
    plt.legend()
    plt.setp(plt.gca().get_legend().get_texts())
    plt.savefig(filepath_val+'geneCall_overDapi_all.png')

def display_spot2gene(gene_ifov, genenames, image_ifov, xdim, ydim):
    IMG = np.zeros((1, 4, xdim, ydim), np.float32)

    for i in range(len(genenames)):
        gene_pos = gene_ifov[gene_ifov['Names'] == genenames[i]]
        for ind in gene_pos.index:
            ipix = (gene_pos['m1'][ind], gene_pos['m2'][ind])
            ran = np.random.rand()*1000
            x = [ipix[0]]
            y = [ipix[1]]
            IMG[:,0,x,y] = np.random.rand()*1000
            IMG[:,1,x,y] = IMG[:,0,x,y]*0.5
            IMG[:,2,x,y] = IMG[:,0,x,y]*0.5
            IMG[:,3,x,y] = IMG[:,0,x,y]*0.5
            # IMG[:,1,ipix[0],ipix[1]] = np.random.rand()*1000
            # IMG[:,2,ipix[0],ipix[1]] = np.random.rand()*1000
            # IMG[:,3,ipix[0],ipix[1]] = np.random.rand()*1000
            # IMG[:,0,ipix[0]-1:ipix[0]+1,ipix[1]-1:ipix[1]+1] = np.random.rand()*1000
            # IMG[:,1,ipix[0]-1:ipix[0]+1,ipix[1]-1:ipix[1]+1] = np.random.rand()*1000
            # IMG[:,2,ipix[0]-1:ipix[0]+1,ipix[1]-1:ipix[1]+1] = np.random.rand()*1000
            # IMG[:,3,ipix[0]-1:ipix[0]+1,ipix[1]-1:ipix[1]+1] = np.random.rand()*1000
    IMG = (IMG).astype(np.uint8)
    output = np.append(IMG, image_ifov, axis=0)
    return output

def save_boundaries(mask_trimmed, positions_reg, config):
    output_all = pd.DataFrame()
    for FOV in range(len(mask_trimmed)):
        mask_fov = mask_trimmed[int(FOV)]
        for ID in range(np.max(mask_fov)):
            ID += 1
            x, y = np.where(mask_fov == int(ID))
            temp = np.zeros((2304, 2304))
            temp[x, y] = str(ID) + '_' + str(FOV)
            outlines = utils.masks_to_outlines(temp)
            x, y = np.nonzero(outlines)
            cell_id = str(ID) + '_' + str(FOV)
            glob_x = [int(i) for i in x*(config.dim/config.xdim)+positions_reg['X'][FOV]-config.dim/2]
            glob_y = [int(i) for i in np.max(positions_reg['Y'])-positions_reg['Y'][FOV]+(y*(config.dim/config.ydim))]
            output = pd.DataFrame(np.transpose([[cell_id] * len(x), x, y, glob_x, glob_y]), columns=['cell_id', 'fov_x', 'fov_y', 'glob_x', 'glob_y'])
            output_all = pd.concat([output_all, output])
    output_all.to_csv(config.filepath_output+'output_all.csv')
    #test_fov.close()

    if not os.path.exists(config.filepath_output + "cell_boundaries/"):
            os.makedirs(config.filepath_output + "cell_boundaries/")
    for FOV in range(len(mask_trimmed)):
        mask_fov = mask_trimmed[int(FOV)]
        test_fov = h5py.File(config.filepath_output + "cell_boundaries/feature_data_" + str(FOV) + ".hdf5", "w")
        for ID in range(np.max(mask_fov)):
            ID += 1
            temp = np.zeros((2304, 2304))
            x, y = np.where(mask_fov == int(ID))
            temp[x, y] = str(ID) + '_' + str(FOV)
            outlines = utils.masks_to_outlines(temp)
            x, y = np.nonzero(outlines)
            glob_x = [int(i) for i in x*(config.dim/config.xdim)+positions_reg['X'][FOV]-config.dim/2]
            glob_y = [int(i) for i in np.max(positions_reg['Y'])-positions_reg['Y'][FOV]+(y*(config.dim/config.ydim))]
            rearrange_x = glob_x[0::2]
            rearrange_x.extend(glob_x[1::2][::-1])
            rearrange_y = glob_y[0::2]
            rearrange_y.extend(glob_y[1::2][::-1])
            data_array = np.column_stack((rearrange_x, rearrange_y))
            cell_id = str(ID) + '_' + str(FOV)
            test_fov.create_group("/featuredata/" + str(cell_id) + "/zIndex_0/p_0/")
            test_fov.create_dataset("/featuredata/" + str(cell_id) + "/zIndex_0/p_0/coordinates", data=(data_array)) # np.array((x, y)).T
        test_fov.close()