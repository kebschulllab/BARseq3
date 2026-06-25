"""
positions correction(align fovs), positions registration(grid stitching)
alignment(channel, round), color-bleed correction, normalization
crop images, apply cropeed masks
"""
import os
import bardensr
import numpy as np
import pandas as pd
import copy
import math
import tifffile
import cv2
from PIL import Image
from datetime import datetime
from n2v.models import N2V
import tensorflow as tf
from itertools import product
from scipy.ndimage import affine_transform
from scipy.signal import correlate2d
# print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"")

# import tensorflow as tf
# import matplotlib.pyplot as plt
# from skimage import io
# from skimage.color import rgb2gray, img_as_float, data, restoration
# from skimage.data import stereo_motorcycle, vortex
# from skimage.transform import warp
# from skimage.registration import optical_flow_tvl1, optical_flow_ilk
# from skimage.restoration import rolling_ball
# import IPython.display
# import SimpleITK as sitk
# import sitkibex
# from pystackreg import StackReg


import cv2

def shading_correction(rounds, filepath_shade_corr):
    print("Performing shading correction....")
    shade_img = tifffile.imread(filepath_shade_corr)
    for i in range(len(rounds)):
        for j in range(len(rounds[0])):
            rounds[i][j] = rounds[i][j] / (shade_img/np.max(shade_img))
            print("FOV"+str(i))
            print("Round"+str(j))
    return rounds
'''
#BARseq3
def colorbleed_correction_inversemat(rounds_aligned, colormixing_matrix=np.array([
    [1,0.6,0,0],  # matrix[0,1] is positive because whenever frame 0 is bright, frame 1 is also a bit bright
    [0.5,1,0.7,0.7],
    [0.12,0.08,1,0.05],
    [0.1,0.137,0.7,1]])):
    fix=np.linalg.inv(colormixing_matrix)

    for i in range(len(rounds_aligned)):
        print("FOV: "+str(i))
        rounds_aligned[i] = np.clip(np.einsum('rcxy,cd->rdxy',rounds_aligned[i],fix),0,None)

    return rounds_aligned
'''
#BARseq3 gene only

def colorbleed_correction_inversemat(rounds_aligned, colormixing_matrix=np.array([
    [1,0.5,0,0],  # matrix[0,1] is positive because whenever frame 0 is bright, frame 1 is also a bit bright
    [0.1,1,0,0],
    [0.12,0.08,1,0.05],
    [0.1,0.137,0.3,1]])):
    fix=np.linalg.inv(colormixing_matrix)

    for i in range(len(rounds_aligned)):
        print("FOV: "+str(i))
        rounds_aligned[i] = np.clip(np.einsum('rcxy,cd->rdxy',rounds_aligned[i],fix),0,None)

    return rounds_aligned

''':
    fix=np.linalg.inv(colormixing_matrix)

    for i in range(len(rounds_aligned)):
        print("FOV: "+str(i))
        rounds_aligned[i] = np.clip(np.einsum('rcxy,cd->rdxy',rounds_aligned[i],fix),0,None)

    return rounds_aligned
'''
'''
#For BARseq2
def colorbleed_correction_inversemat(rounds_aligned, colormixing_matrix=np.array([
    [1,0.6,0,0],  # matrix[0,1] is positive because whenever frame 0 is bright, frame 1 is also a bit bright
    [0.5,1,0.0,0.0],
    [0.0,0.0,1,0.15],
    [0.0,0.0,0.7,1]])):
    fix=np.linalg.inv(colormixing_matrix)

    for i in range(len(rounds_aligned)):
        print("FOV: "+str(i))
        rounds_aligned[i] = np.clip(np.einsum('rcxy,cd->rdxy',rounds_aligned[i],fix),0,None)

    return rounds_aligned
'''
"""
def colorbleed_correction_inversemat(rounds_aligned, colormixing_matrix=np.array([
    [1,0.6,0,0],  # matrix[0,1] is positive because whenever frame 0 is bright, frame 1 is also a bit bright
    [0.5,1,0.0,0.0],
    [0.0,0.0,1,0.15],
    [0.0,0.0,0.7,1]])):
    fix=np.linalg.inv(colormixing_matrix)

    for i in range(len(rounds_aligned)):
        print("FOV: "+str(i))
        rounds_aligned[i] = np.clip(np.einsum('rcxy,cd->rdxy',rounds_aligned[i],fix),0,None)

    return rounds_aligned
"""
def align_fovs(positions, dim):
    dimensions = dim * 0.9
    xpos = (positions['X'] - np.min(positions['X'])).astype(int)
    ypos = (positions['Y'] - np.min(positions['Y'])).astype(int)
   
    num_col = int((np.max(xpos) - np.min(xpos)) / dimensions) + 2
    num_row = int((np.max(ypos) - np.min(ypos)) / dimensions) + 2

    xgrid = np.zeros(len(xpos), dtype=int)
    ygrid = np.zeros(len(ypos), dtype=int)
    shift = 100

    for i in range(num_col):
        for j in range(num_row):
            x_start = i * dimensions - (j + 1) * shift
            x_end = i * dimensions + (j + 1) * shift
            y_start = j * dimensions - (i + 1) * shift
            y_end = j * dimensions + (i + 1) * shift

            xgrid[(xpos >= x_start) & (xpos < x_end)] = i
            ygrid[(ypos >= y_start) & (ypos < y_end)] = j

    xpos_corr = xgrid * dimensions
    ypos_corr = -ygrid * dimensions

    positions_corr = pd.DataFrame({
        'X': xpos_corr - np.min(xpos_corr) + dim // 2,
        'Y': ypos_corr - np.min(ypos_corr) + dim // 2
    }).astype(int)

    return positions_corr

def grid_stitch(ij, dapi_images, positions_corr, xdim, ydim, filepath_gridstch):
    fov_num = len(dapi_images)
    for i in range(fov_num):
        merge_im = Image.fromarray(np.uint8(dapi_images[i]))
        merge_im.save(filepath_gridstch + 'tile'+str(i)+'.tif')
    # use positions_corr for grid stitching (y is flipped)
    # positions_corr has appended zero matrixs' coordinates
    positions = pd.DataFrame()
    ypos_corr = [-1*i for i in positions_corr['Y']]
    positions['X'] = positions_corr['X']  - np.min(positions_corr['X'] )
    positions['Y'] = ypos_corr - np.min(ypos_corr)
    # prepare positions.txt in correct format for grid stitching
    with open(filepath_gridstch + 'positions.txt', 'w') as f:
        f.write('dim = 2 \n')
        for i in range(len(positions)):
            if i == fov_num:
                break
            f.write('tile'+str(i)+'.tif; ; ('+str(positions['Y'][i])+', '+str(positions['X'][i])+')\n')
    f.close()

    # grid stitching by ImageJ: run stitch_images.ijm integration
    script_file = open("starmap/stitch_images.ijm", "r")
    stitch_images_script = script_file.read()
    script_file.close()
    # run script
    args = {'filepath': filepath_gridstch}
    result = ij.py.run_script("ijm", stitch_images_script, args)

    # read in registered positions and saved in .csv
    xpos_reg, ypos_reg = [], []
    with open(filepath_gridstch + 'positions.registered.txt', 'r') as f:
        for line in f:
            if "tile" in line:
                ypos_reg.append(float(line.split('(')[1].split(', ')[0])*-1)
                xpos_reg.append(float(line.split(', ')[1].split(')')[0]))
    positions_reg = pd.DataFrame()
    positions_reg['X'] = xpos_reg - np.min(xpos_reg) + xdim/2
    positions_reg['Y'] = ypos_reg - np.min(ypos_reg) + ydim/2
    
    return positions_reg

### NEW Function for Channel Alignment with OpenCV
def channel_alignment_opencv(images, fov_align): # assumes structure is fov x round x channel x X x Y
    # Align channels by ImageJ through running align_channels.ijm
    # read script for channel alignment
    transforms = []
    for i in range(len(images[0])):
        print("finding channel alignment for round "+ str(i))
        c2 = images[fov_align][ i, 2]
        #c2 = np.expand_dims(np.stack([c2,c2,c2,c2], axis = 0), axis = 1)
        print(c2.shape)
        #c2 = bardensr.preprocessing.background_subtraction((c2/np.max(c2)).astype('float'),[0,50,50])
        #c2 = np.sum(c2[:,0], axis = 0)
        c2 = np.clip(c2, 0, np.percentile(c2.ravel(), 99.5))
        c2 = (255-(c2/np.max(c2))*255).astype('uint8')

        c3 = images[fov_align][ i, 3]
        #c3 = np.expand_dims(np.stack([c3,c3,c3,c3], axis = 0), axis = 1)
        print(c3.shape)
        #c3 = bardensr.preprocessing.background_subtraction((c3/np.max(c3)).astype('float'),[0,50,50])
        #c3 = np.sum(c3[:,0], axis = 0)
        c3 = np.clip(c3, 0, np.percentile(c3.ravel(), 99.5))
        c3 = (255-(c3/np.max(c3))*255).astype('uint8')
        print(c2.shape)
        print(c3.shape)
        # Detect keypoints and compute descriptors
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.blobColor = 0 # lighter blobs
        params.minThreshold = 5
        params.maxThreshold = 250

        # Filter by Area.
        params.filterByArea = True
        params.maxArea = 100000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints1 = detector.detect(c3)
        keypoints2 = detector.detect(c2)

        # Convert keypoints to numpy arrays of points
        # Convert keypoints to numpy arrays of points
        pts1 = np.float32([kp.pt for kp in keypoints1])#.reshape(-1, 1, 2)
        pts2 = np.float32([kp.pt for kp in keypoints2])#.reshape(-1, 1, 2)

        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors
        matches = bf.match(pts1, pts2)
        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance)
        
        print(len(matches))
        # Remove not so good matches
        num_good_matches = int(len(matches) * 1)
        matches = matches[:num_good_matches]


        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        transforms.append(H)
    
    for i in range(len(images[0])):
        for j in range(len(images)):
            print("applying channel alignment for round "+ str(j))

            images[j][ i, 3] = cv2.warpPerspective(images[j][ i, 3], transforms[i], (images[j][ i, 3].shape[1], images[j][ i, 3].shape[0]))
            images[j][ i, 1] = cv2.warpPerspective(images[j][ i, 1], transforms[i], (images[j][ i, 1].shape[1], images[j][ i, 1].shape[0]))
    return images

### NEW Function for Round Alignment with OpenCV
def round_alignment_opencv(rounds, round_index, filepath_val, round_align): # assumes structure is fov x round x channel x X x Y
    # (AFTER BACKGROUND SUBTRACTION)
    # Detect keypoints and compute descriptors
    rounds = np.stack(rounds, axis = 0)[:, round_index] #added

    images = copy.deepcopy(rounds)
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.blobColor = 0 # lighter blobs
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.maxArea = 10000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    transforms = []
    for i in range(len(images)):
        r1 = np.expand_dims(images[i][round_align],axis = 0)
        r1 = bardensr.preprocessing.background_subtraction((r1/np.max(r1)).astype('float'),[0,10,10])
        r1 = np.sum(r1[0], axis = 0)
        r1 = np.clip(r1, 0, np.percentile(r1.ravel(), 95))
        r1 = (255-(r1/np.max(r1))*255).astype('uint8')
        r11 = r1[10:2250, 10:2250]
        for j in range(len(images[0])):
            print("round alignment for fov "+str(i)+" and round "+ str(j))
            rn = np.expand_dims(images[i][ j],axis = 0)
            rn = bardensr.preprocessing.background_subtraction((rn/np.max(rn)).astype('float'),[0,10,10])
            rn = np.sum(rn[0], axis = 0)
            rn = np.clip(rn, 0, np.percentile(rn.ravel(), 95))
            rn = (255-(rn/np.max(rn))*255).astype('uint8')
            rn1 = rn[10:2250, 10:2250]
            # Detect blobs
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(rn1,None)
            kp2, des2 = sift.detectAndCompute(r11,None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50) # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            # Need to draw only good matches, so create a mask
            #matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            good = []
            for p,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    good.append(m)
            # Sort matches by score
            #matches = sorted(matches, key=lambda x: x.distance)
            print('Number of matches between round 1 and ' + str(i)+": "+ str(len(good)))
            #img3 = cv2.drawMatches(r11,kp1,rn1,kp2,good[:5],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #plt.imshow(img3),plt.show()
            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # Find homography
            H, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            print(i)
            print(j)
            print(images[i][ j, 0].shape)
            images[i][ j, 0] = cv2.warpAffine(images[i][ j, 0], H, (images[i][ j, 0].shape[1], images[i][ j, 0].shape[0]))
            images[i][ j, 1] = cv2.warpAffine(images[i][ j, 1], H, (images[i][ j, 1].shape[1], images[i][ j, 1].shape[0]))
            images[i][ j, 2] = cv2.warpAffine(images[i][ j, 2], H, (images[i][ j, 2].shape[1], images[i][ j, 2].shape[0]))
            images[i][ j, 3] = cv2.warpAffine(images[i][ j, 3], H, (images[i][ j, 3].shape[1], images[i][ j, 3].shape[0]))     
        import PIL
        imgs = [bardensr.plotting.lutup(*x,sc=.5,normstyle='each') for x in images[i][:,:]]
        tifffile.imsave(filepath_val+'registration_ralign_QC_fov'+str(i)+'.tif', np.array(imgs)) # could be opened with IFJI or napari
        imgs = [PIL.Image.fromarray(x) for x in imgs]
        imgs[0].save(filepath_val+'registration_ralign_QC_fov'+str(i)+'.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0) # 
    return images

def channel_alignment(ij, round_num, fov_align, filepath_imageJ, filepath_chalign):
    # Align channels by ImageJ through running align_channels.ijm
    # read script for channel alignment
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"channel alignment")
    script_file = open(filepath_chalign, "r")
    align_channel_script = script_file.read()
    script_file.close()
    # run script
    args = {'round_num': round_num, 'fov_align': fov_align,'filepath': filepath_imageJ}
    result = ij.py.run_script("ijm", align_channel_script, args)

def round_alignment(ij, round_num, round_align, filepath_imageJ, filepath_roalign):
    # Align rounds by ImageJ through running align_rounds.ijm
    # read script for round alignment
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"round alignment")
    script_file = open(filepath_roalign, "r")
    align_round_script = script_file.read()
    script_file.close()
    # run script
    args = {'round_num': round_num, 'round_align': round_align, 'filepath': filepath_imageJ}
    result = ij.py.run_script("ijm", align_round_script, args)

def dapi_alignment(ij, fov_num, dapi_round, filepath_imageJ, subfolder):
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"alignment for "+subfolder)
    script_file = open("starmap/align_dapi.ijm", "r")
    align_dapi_script = script_file.read()
    script_file.close()
    # run script
    args = {'fov_num': fov_num, 'dapi_round': dapi_round, 'filepath': filepath_imageJ, 'subfolder': subfolder}
    result = ij.py.run_script("ijm", align_dapi_script, args)

def get_border_size(image, x_dim, y_dim):
    data_temp = image.reshape(image.shape[0]*image.shape[1], x_dim, y_dim)
    mask = np.ones([x_dim, y_dim])

    for i in range(len(data_temp)):
        mask = mask*data_temp[i] != 0

    nonzero_coords = np.argwhere(mask != 0)
    top, left = nonzero_coords.min(axis=0)
    bottom, right = nonzero_coords.max(axis=0)

    cropped_image = data_temp[:,top+20:bottom-20, left+20:right-20].astype(np.float64)
    mask_fov = np.zeros([x_dim, y_dim])
    mask_fov[top+20:bottom-20, left+20:right-20] = 1 # clip 20 pixels on each side to account for rotation
    data_temp = cropped_image.reshape(image.shape[0],image.shape[1],cropped_image.shape[1],cropped_image.shape[2])

    return data_temp, mask_fov

import numpy as np

# Keep the image's relative position, but pad to the reference size
def get_border_size_1(image, x_dim, y_dim):
    # (rounds, channels, H, W) → (rounds*channels, H, W)
    data_temp = image.reshape(image.shape[0]*image.shape[1], x_dim, y_dim)
    
    # mask：所有轮次都非零的区域
    mask = np.ones((x_dim, y_dim), dtype=bool)
    for i in range(len(data_temp)):
        mask &= (data_temp[i] != 0)
    
    # 找到非零区域的边界
    nonzero_coords = np.argwhere(mask)
    top, left = nonzero_coords.min(axis=0)
    bottom, right = nonzero_coords.max(axis=0)

    # 用第二轮的图像作为基准大小
    H_ref, W_ref = image[1].shape[-2], image[1].shape[-1]

    # 裁剪后的区域
    cropped_image = data_temp[:, top:bottom+1, left:right+1].astype(np.float64)

    # 初始化填充后的结果
    padded = np.zeros((data_temp.shape[0], H_ref, W_ref), dtype=np.float64)
    mask_fov = np.zeros((H_ref, W_ref), dtype=np.uint8)

    # 把裁剪的图像放回 padded 的对应位置（保持原始 top,left 偏移）
    h, w = cropped_image.shape[-2:]
    padded[:, top:top+h, left:left+w] = cropped_image
    mask_fov[top:top+h, left:left+w] = 1

    # reshape 回 (rounds, channels, H_ref, W_ref)
    data_temp = padded.reshape(image.shape[0], image.shape[1], H_ref, W_ref)

    return data_temp, mask_fov

def crop_images(image_matrix, x_dim, y_dim):
    data_align_crop = []
    masks = []
    # Find the maximum border size among all images
    for image in image_matrix:
        data_temp,mask = get_border_size(image, x_dim, y_dim)
        data_align_crop.append(data_temp)
        masks.append(mask)    
    return data_align_crop,masks
    
# colorbleed correction    
def colorbleed_correction(rounds_aligned, cc_coeff = [0.05, 0.45, 0.2, 0.05]):
    [zero_one, one_zero, two_three, three_two] = cc_coeff
    rounds_cb = copy.deepcopy(rounds_aligned)
    for i in range(len(rounds_aligned)):
        for j in range(len(rounds_aligned[0])):
            rounds_cb[i][j, 0] = rounds_cb[i][j, 0] - zero_one * rounds_aligned[i][j, 1]
            rounds_cb[i][j, 1] = rounds_cb[i][j, 1] - one_zero * rounds_aligned[i][j, 0]
            rounds_cb[i][j, 2] = rounds_cb[i][j, 2] - two_three * rounds_aligned[i][j, 3]
            rounds_cb[i][j, 3] = rounds_cb[i][j, 3] - three_two * rounds_aligned[i][j, 2]
    return rounds_cb

def minmax(rounds, fovs, precision, clip): # precision: 0.1/0.0005
    min_values, max_values = [], []
    # Iterate over each FOV
    for i in fovs:
        # Compute the minimum and maximum values along each (round * channel) dimensions for current FOV
        min_value = np.percentile(rounds[i], precision, axis=(1,2), keepdims=True) # np.percentile(rounds_cb[j], axis=(1, 2),keepdims=True)
        max_value = np.percentile(rounds[i], 100-precision, axis=(1,2), keepdims=True) # tf.reduce_max(rounds_cb[j], axis=(1, 2),keepdims=True)
        min_values.append(min_value)
        max_values.append(max_value)
    # Convert the lists to TensorFlow tensors
    min_values_all = np.stack(min_values, axis=0)
    max_values_all = np.stack(max_values, axis=0)

    # Compute the median of the min and max values across all FOVs
    median_min = np.median(min_values_all, axis=0)
    median_max = np.median(max_values_all, axis=0)
    if clip:
        for i in range(len(median_max)):
            temp_max = max_values_all[:,i,0,0]
            while median_max[i] > 4000:
                temp_max = np.delete(np.sort(temp_max), -1)
                median_max[i] = np.median(temp_max)
            if math.isnan(median_max[i]):
                median_max[i] = 1000

    return median_min, median_max

def apply_n2v(images, model_path, round_num):
    # load pre-trained model
    model_c0 = N2V(config=None, name='chan0', basedir=model_path)
    model_c1 = N2V(config=None, name='chan1', basedir=model_path)
    model_c2 = N2V(config=None, name='chan2', basedir=model_path)
    model_c3 = N2V(config=None, name='chan3', basedir=model_path)
    model_dic = {0:model_c0, 1:model_c1, 2:model_c2, 3:model_c3}
    # apply model
    result = []
    for fov in images:
        imgs = copy.deepcopy(fov)
        for i in range(round_num):
            for j in range(4):
                origin = np.swapaxes([imgs[i, j, :, :]], 2, 0)
                predict = model_dic[j].predict(origin, axes='YXC')
                imgs[i, j] = np.swapaxes(predict, 2, 0)
        result.append(imgs)
    return result

# image normalization
def round_normalization(rounds, fovs, radius): # minmax-rb-n2v(*option)-minmax
    R = rounds[0].shape[0]
    C = rounds[0].shape[1]
    #rounds = copy.deepcopy(rounds)
    # for i in range(len(rounds)): # clip images to remove outlier pixel values
    #     rounds[i] = np.clip(rounds[i],10,4000)
    for i in range(len(rounds)): # flat each fov (round*channel, xdim, ydim)
        rounds[i] = rounds[i].reshape(R*C, rounds[i].shape[2], rounds[i].shape[3])
    # step 1. min-max normalization
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 1. min-max normalization")
    median_min, median_max = minmax(rounds, fovs, 0.1, True)
    rounds = [(rc - median_min) / median_max for rc in rounds]
    for i in range(len(rounds)):
        rounds[i] = np.clip(rounds[i],0,1)
    # step 2. background subtraction
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 2. background subtraction")
    with tf.device('/device:GPU:0'):
        for i in range(len(rounds)):
            print("FOV: "+str(i))
            temp = bardensr.preprocessing.background_subtraction(rounds[i].reshape(R,C,rounds[i].shape[1], rounds[i].shape[2]).astype(np.float64), [0,radius,radius])
            rounds[i] = temp.reshape(R*C, rounds[i].shape[1],rounds[i].shape[2])
    # # step *. n2v denosing
    # print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step *. n2v denosing")
    # for i in range(len(rounds)): # reshape each fov (round, channel, xdim, ydim)
    #     rounds[i] = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2])
    # rounds = apply_n2v(rounds, 'models/MA04_A1_RB', R)
    # for i in range(len(rounds)): # flat each fov (round*channel, xdim, ydim)
    #     rounds[i] = images[i].reshape(R*C, images[i].shape[2], images[i].shape[3])
    # step 3. min-max normalization
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 3. min-max normalization")
    median_min, median_max = minmax(rounds, fovs, 0.0005, False)
    # median_max[median_max == 0] = 1 #Chatgpt suggestion to avoid division by zero HH
    #rounds = [(rc - median_min) / median_max for rc in rounds] #original version

    # 安全版 min-max normalization，避免除零和 NaN
    safe_rounds = []
    eps = 1e-6  # 防止除以零的小常数
    for rc in rounds:
        # 保证 median_max 没有 0
        safe_max = np.where(median_max == 0, eps, median_max)
        # 做归一化
        norm_img = (rc - median_min) / safe_max
        # 清理 NaN / inf
        norm_img = np.nan_to_num(norm_img, nan=0.0, posinf=0.0, neginf=0.0)
        safe_rounds.append(norm_img)

    rounds = safe_rounds    
#chatgpt above

    for i in range(len(rounds)): # clip images to remove outlier pixel values
        rounds[i] = np.clip(rounds[i],0,1)
    for i in range(len(rounds)): # reshape each fov (round, channel, xdim, ydim)
        rounds[i] = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2])
    return rounds


"""
def round_normalization(rounds, fovs, radius,
                        # Step1/2 仍用你现在的设置
                        p_low=2.0, p_high=99.5,
                        # ---- 新增：小点锐化 & 高亮抑制参数 ----
                        sharp_sigma_small=0.5,   # 小尺度 σ（越小越偏向更细小的点）
                        sharp_sigma_large=2.0,   # 大尺度 σ（越大越抑制大结构）
                        sharp_alpha=0.5,         # 小点增益（0.25~0.6 常用）
                        cap_pct=99.9,            # 亮度温和截顶的分位（先温柔去极端）
                        knee_pct=90.0,           # 软膝阈值（>knee 的高亮开始被压）
                        knee_softness=0.20,      # 软膝柔度（0.1~0.3；越大越柔和）
                        gamma=0.95,              # 最后轻度γ（<1 提亮点，别太小）
                        use_dog=True
                        ):
    """
""""
    rounds: list of FOV arrays, 每个元素形状 (R, C, H, W)
    fovs:   代表性 FOV 索引（如 [2]）
    radius: 背景扣除半径
    """
"""
    import numpy as np, cv2, bardensr, tensorflow as tf
    from datetime import datetime

    # ---------- 展平成 (R*C, H, W) ----------
    R, C = rounds[0].shape[0], rounds[0].shape[1]
    for i in range(len(rounds)):
        rounds[i] = rounds[i].reshape(R*C, rounds[i].shape[2], rounds[i].shape[3]).astype(np.float32)

    # ---------- Step 1: 稳健百分位拉伸（同你现在的做法） ----------
    def compute_anchors(arr_rc):
        lo = np.percentile(arr_rc, p_low)
        hi = np.percentile(arr_rc, p_high)
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        return lo, hi

    anchors, eps = [], 1e-6
    for rc in range(R*C):
        lows, highs = [], []
        for fi in fovs:
            lo, hi = compute_anchors(rounds[fi][rc])
            lows.append(lo); highs.append(hi)
        anchors.append((np.median(lows), np.median(highs)))

    for i in range(len(rounds)):
        rc_stack, out = rounds[i], np.empty_like(rounds[i], dtype=np.float32)
        for rc in range(R*C):
            lo, hi = anchors[rc]; x = rc_stack[rc]
            x = (x - lo) / max(hi - lo, eps)
            out[rc] = np.clip(x, 0, 1)
        rounds[i] = out

    # ---------- Step 2: 背景扣除（保持原样） ----------
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        step 2. background subtraction")
    with tf.device('/device:GPU:0'):
        for i in range(len(rounds)):
            x = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2]).astype(np.float64)
            x = bardensr.preprocessing.background_subtraction(x, [0, radius, radius])
            rounds[i] = x.reshape(R*C, x.shape[2], x.shape[3]).astype(np.float32)

    # ---------- Step 3: 小点“限幅锐化” ----------
    # 先 DoG 得到小尺度带通，再归一化后以 sharp_alpha 融合；融合前对 DoG 做 [0,1] 归一化，保证弱小点也能被提起来
    if use_dog:
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        step 3. bandpass sharpen for dots")
        for i in range(len(rounds)):
            rc_stack = rounds[i]
            for rc in range(R*C):
                img = rc_stack[rc]
                b1  = cv2.GaussianBlur(img, (0, 0), sharp_sigma_small)
                b2  = cv2.GaussianBlur(img, (0, 0), sharp_sigma_large)
                band = np.clip(b1 - b2, 0, None)
                m = band.max()
                if m > 0:
                    band = band / m
                # 限幅锐化：避免把巨亮斑继续拉爆
                rc_stack[rc] = np.clip(img + sharp_alpha * band, 0, 1)
            rounds[i] = rc_stack

    # ---------- Step 4: 亮点抑制（温和截顶 + 软膝）+ 轻 γ ----------
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        step 4. highlight suppression (cap + soft-knee) + mild gamma")

    def soft_knee(x, knee=0.9, softness=0.2):
        # x ∈ [0,1]；knee_softness 是相对 (1-knee) 的比例
        x = np.clip(x, 0, 1)
        if softness <= 0:  # 无软膝则直接返回
            return x
        denom = max(softness * (1.0 - knee), 1e-6)
        above = x - knee
        y = np.where(x <= knee, x, knee + (1.0 - (np.exp(-(above / denom)))) * (1.0 - knee))
        return np.clip(y, 0, 1)

    for i in range(len(rounds)):
        rc_stack = rounds[i]
        out = np.empty_like(rc_stack, dtype=np.float32)
        for rc in range(R*C):
            x = rc_stack[rc]

            # (1) 温和截顶：把极少数超亮像素压回 cap 分位
            cap = np.percentile(x, cap_pct)
            if np.isfinite(cap) and cap > 0:
                x = np.minimum(x, cap) / max(cap, eps)
            x = np.clip(x, 0, 1)

            # (2) 软膝：只对 knee 以上的亮部做柔性压缩
            knee = np.percentile(x, knee_pct)
            knee = float(np.clip(knee, 0.5, 0.98))  # 保证阈值合理
            y = soft_knee(x, knee=knee, softness=knee_softness)

            # (3) 轻度 γ（<1 稍微抬点，别太小）
            if gamma is not None and gamma < 1.0 and gamma > 0:
                y = np.power(y, gamma)

            out[rc] = np.clip(np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)

        rounds[i] = out

    # ---------- reshape 回原状 ----------
    for i in range(len(rounds)):
        rounds[i] = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2])
    return rounds
"""
'''
def round_normalization(rounds, fovs, radius): # minmax-rb-n2v(*option)-minmax
    R = rounds[0].shape[0]
    C = rounds[0].shape[1]
    #rounds = copy.deepcopy(rounds)
    # for i in range(len(rounds)): # clip images to remove outlier pixel values
    #     rounds[i] = np.clip(rounds[i],10,4000)
    for i in range(len(rounds)): # flat each fov (round*channel, xdim, ydim)
        rounds[i] = rounds[i].reshape(R*C, rounds[i].shape[2], rounds[i].shape[3])
    # step 1. min-max normalization
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 1. min-max normalization")
    median_min, median_max = minmax(rounds, fovs, 0.1, True)
    rounds = [(rc - median_min) / median_max for rc in rounds]
    for i in range(len(rounds)):
        rounds[i] = np.clip(rounds[i],0,1)
    # step 2. background subtraction
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 2. background subtraction")
    with tf.device('/device:GPU:0'):
        for i in range(len(rounds)):
            print("FOV: "+str(i))
            temp = bardensr.preprocessing.background_subtraction(rounds[i].reshape(R,C,rounds[i].shape[1], rounds[i].shape[2]).astype(np.float64), [0,radius,radius])
            rounds[i] = temp.reshape(R*C, rounds[i].shape[1],rounds[i].shape[2])
    # # step *. n2v denosing
    # print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step *. n2v denosing")
    # for i in range(len(rounds)): # reshape each fov (round, channel, xdim, ydim)
    #     rounds[i] = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2])
    # rounds = apply_n2v(rounds, 'models/MA04_A1_RB', R)
    # for i in range(len(rounds)): # flat each fov (round*channel, xdim, ydim)
    #     rounds[i] = images[i].reshape(R*C, images[i].shape[2], images[i].shape[3])
    # step 3. min-max normalization
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 3. min-max normalization")
    median_min, median_max = minmax(rounds, fovs, 0.0005, False)
    rounds = [(rc - median_min) / median_max for rc in rounds]
    for i in range(len(rounds)): # clip images to remove outlier pixel values
        rounds[i] = np.clip(rounds[i],0,1)
    for i in range(len(rounds)): # reshape each fov (round, channel, xdim, ydim)
        rounds[i] = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2])
    return rounds
'''
def round_normalization_n2v(images, fovs, model_dic): # minmax-n2v-minmax
    R = images[0].shape[0]
    C = images[0].shape[1]
    rounds = copy.deepcopy(images)
    for i in range(len(rounds)): # clip images to remove outlier pixel values
        rounds[i] = np.clip(rounds[i], 10, 4000)
    for i in range(len(rounds)): # flat each fov (round*channel, xdim, ydim)
        rounds[i] = images[i].reshape(R*C, images[i].shape[2], images[i].shape[3])
    # step 1. min-max normalization
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 1. min-max normalization")
    median_min, median_max = minmax(rounds, fovs, 0.1, True)
    rounds = [(rc-median_min) / (median_max-median_min) for rc in rounds]
    for i in range(len(rounds)): # clip images to remove outlier pixel values
        rounds[i] = np.clip(rounds[i], 0, 1)
    # step 2. n2v denosing
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step *. n2v denosing")
    for i in range(len(rounds)): # reshape each fov (round, channel, xdim, ydim)
        rounds[i] = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2])
    rounds = apply_n2v(rounds, 'models/MA04_A1', R)
    for i in range(len(rounds)): # flat each fov (round*channel, xdim, ydim)
        rounds[i] = images[i].reshape(R*C, images[i].shape[2], images[i].shape[3])
    # step 3. min-max normalization
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"step 3. min-max normalization")
    median_min, median_max = minmax(rounds, fovs, 0.0005, False)
    rounds = [(rc-median_min) / (median_max-median_min) for rc in rounds]
    for i in range(len(rounds)): # clip images to remove outlier pixel values
        rounds[i] = np.clip(rounds[i], 0, 1)

    for i in range(len(rounds)): # reshape each fov (round, channel, xdim, ydim)
        rounds[i] = rounds[i].reshape(R, C, rounds[i].shape[1], rounds[i].shape[2])

    return rounds

def apply_mask_to_images(image_array, mask):
    mask_width, mask_height = mask.shape
    x_index, y_index = np.where(mask == 1)
    x_st, x_ed, y_st, y_ed = min(x_index), max(x_index), min(y_index), max(y_index)
    result_array = np.zeros((image_array.shape[0], image_array.shape[1], mask_width, mask_height), dtype=image_array.dtype)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            x_core, y_core = image_array[i, j].shape[0], image_array[i, j].shape[1]
            if (x_ed-x_st+1) != x_core or (y_ed-y_st+1) != y_core:
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"masks not fit!")
            resized_image = np.zeros((mask_width, mask_height), dtype=image_array.dtype)
            resized_image[x_st:x_ed+1, y_st:y_ed+1] = image_array[i, j]
            result_array[i, j] = resized_image
            
    return result_array
''' # original registration
def bardensr_registration(images, codeflat, x_dim, y_dim):
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----bardensr registration")
    R = images[0].shape[0]
    C = images[0].shape[1]
    rounds = copy.deepcopy(images)
    for i in range(len(rounds)): # clip images to remove outlier pixel values
        Xflat = rounds[i].reshape(R*C, 1, rounds[i].shape[2], rounds[i].shape[3])
        Xnorm = Xflat#[:, :, 300:2000, 300:2000]
        with tf.device('/device:GPU:0'):
            corrections = bardensr.registration.find_translations_using_model(Xnorm,codeflat,use_tqdm_notebook=True)
            flat, newt = bardensr.registration.apply_translations(Xflat, corrections.corrections)
        reflat = flat.reshape((R,C,)+flat.shape[1:])
        xshape = min(reflat.shape[-2], x_dim)
        yshape = min(reflat.shape[-1], y_dim)
        rounds[i][:, :, :xshape, :yshape] = reflat[:, :, 0, :xshape, :yshape]
    return rounds
'''
# from Manjari
def register_stack_with_demons(
  imagestack,
  reference_index=0,
  niter=150,
  smoothing_sigma=2.0):
  '''
  Demon registration on full-resolution images (no downsampling)
  '''
  R, C, H, W = imagestack.shape
  images_sum = np.sum(imagestack, axis=1)
  # Prepare fixed SITK image
  fixed_np = images_sum[reference_index]
  fixed = to_sitk_image(fixed_np)
  transforms = [None]*R
  aligned_stack = np.zeros_like(imagestack)
  # Reference frame identity
  aligned_stack[reference_index] = imagestack[reference_index]
  identity_tf = sitk.Transform(2, sitk.sitkIdentity)
  transforms[reference_index] = identity_tf
  for i in range(R):
    if i == reference_index:
      continue
    #print(f”Processing frame {i}“)
    moving_np = images_sum[i]
    moving = to_sitk_image(moving_np)
    # Initialize and run Demons registration
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(niter)
    demons.SetStandardDeviations(smoothing_sigma)
    displacement_field = demons.Execute(fixed, moving)
    # print(f”RMS Change: {demons.GetRMSChange():.6f} after {demons.GetElapsedIterations()} iterations”)
    # Convert to displacement field transform
    displacement_field = sitk.DisplacementFieldTransform(displacement_field)
    transforms[i] = displacement_field
    # Resample all channels
    for ch in range(C):
      mov_ch = to_sitk_image(imagestack[i,ch])
      res = sitk.Resample(
        mov_ch, fixed, displacement_field,
        sitk.sitkLinear, 0.0, mov_ch.GetPixelID()
      )
      aligned_stack[i,ch] = from_sitk_image(res)
  return transforms, aligned_stack


def register_stack_with_bspline_downsample(
    imagestack,
    reference_index=0,
    spline_grid=(8, 8),
    niter=10,
    learning_rate=0.5,
    max_disp=10,
    downsample_size=(1000, 1000)
):
    R = imagestack.shape[0]  # Number of frames in the stack

    # Summing across channels for single-channel registration
    images_sum = np.sum(imagestack, axis=1)
    fixed_np = images_sum[reference_index]
    fixed_sitk = to_sitk_image(fixed_np)

    transforms = [None] * R
    aligned_stack = np.zeros_like(imagestack)

    aligned_stack[reference_index] = imagestack[reference_index]
    # Use identity transform as BSpline for reference
    identity_bspline = sitk.BSplineTransformInitializer(
        fixed_sitk, transformDomainMeshSize=spline_grid, order=3
    )
    transforms[reference_index] = identity_bspline

    for i in range(R):
        print("Processing frame", i)
        if i == reference_index:
            continue

        moving_np = images_sum[i]
        moving_sitk = to_sitk_image(moving_np)

        # Initialize a B-spline transform for this registration
        initial_transform = sitk.BSplineTransformInitializer(
            fixed_sitk,
            transformDomainMeshSize=spline_grid,
            order=3,
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsCorrelation()
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=learning_rate,
            numberOfIterations=niter,
        )
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

        # Ensure final_transform is a BSplineTransform
        if isinstance(final_transform, sitk.BSplineTransform):
            final_transform = restrict_control_point_displacement(final_transform, max_disp=max_disp)
        else:
            print(f"Warning: Transform for frame {i} is not a pure BSplineTransform.")
            # Skip displacement restriction for non-BSplineTransform

        transforms[i] = final_transform

        # Resample each channel of the moving frame and stack them in the original order
        resampled_channels = []
        for ch in range(imagestack.shape[1]):  # Loop over channels
            moving_channel_sitk = to_sitk_image(imagestack[i, ch])
            resampled_channel = sitk.Resample(
                moving_channel_sitk,
                fixed_sitk,  # Reference frame for alignment
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_channel_sitk.GetPixelID()
            )
            resampled_channels.append(from_sitk_image(resampled_channel))

        # Stack resampled channels along the channel axis (preserve order)
        aligned_stack[i] = np.stack(resampled_channels, axis=0)

    return transforms, aligned_stack

# Assume imagestack is defined, e.g.:
# imagestack = np.random.rand(10, 256, 256)  # 10 random 2D images of size 256x256

# Use the function to get transforms and aligned images

from bardensr.spot_calling import blackberry

def estimate_density_iterative_fp32(imagestack, codebook, l1_penalty=0, psf_radius=(0,0,0),
                    iterations=100, estimate_codebook_gain=True,
                    rounds=None,
                    estimate_colormixing=False, estimate_phasing=False,
                    use_tqdm_notebook=False):
    """
    Float32-safe version of bardensr's estimate_density_iterative.
    """

    imagestack = imagestack.astype(np.float32)
    codebook = codebook.astype(np.float32)

    F = imagestack.shape[0]
    J = codebook.shape[-1]
    niter = iterations

    if rounds is None:
        rounds = 1
    else:
        assert F % rounds == 0

    imagestack_rc = imagestack.reshape((rounds, F // rounds) + imagestack.shape[1:])
    codebook = codebook.reshape((rounds, F // rounds, codebook.shape[-1]))

    imagestack_rct = tf.convert_to_tensor(
        np.transpose(imagestack_rc, [2, 3, 4, 0, 1]),
        dtype=tf.float32
    )

    M0, M1, M2 = imagestack_rct.shape[:3]

    m = blackberry.denselearner.Model(codebook, (M0, M1, M2),
                                      lam=l1_penalty,
                                      blur_level=psf_radius)

    if use_tqdm_notebook:
        import tqdm.notebook
        t = tqdm.notebook.trange(niter)
    else:
        t = range(niter)

    for i in t:
        m.update_F(imagestack_rct)
        if estimate_codebook_gain:
            m.update_alpha(imagestack_rct)
        if estimate_colormixing:
            m.update_varphi(imagestack_rct)
        if estimate_phasing:
            m.update_rho(imagestack_rct)
        m.update_a(imagestack_rct)
        m.update_b(imagestack_rct)

    return m.F_scaled(), dict(frame_gains=m.alpha.numpy().ravel())

def call_genes_tiled(
    Xflat, codeflat, tile_size=1024, iterations=10, **kwargs
):
    F, _, H, W = Xflat.shape
    assert _ == 1

    J = codeflat.shape[-1]
    result_full = np.zeros((H, W, J), dtype=np.float32)

    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            i_end = min(i + tile_size, H)
            j_end = min(j + tile_size, W)

            # Extract spatial tile from all frames
            Xtile = Xflat[:, :, i:i_end, j:j_end]

            try:
                et_tile, _ = bardensr.spot_calling.estimate_density_iterative(
                    Xtile, codeflat, iterations=iterations, **kwargs
                )
            except Exception as e:
                print(f"Tile ({i}:{i_end}, {j}:{j_end}) failed: {e}")
                continue

            # Insert result into final output
            result_full[i:i_end, j:j_end, :] = et_tile[:i_end - i, :j_end - j, :]

    return result_full

def call_genes_tiled_with_overlap(
    Xflat, codeflat, tile_size=1024, overlap=64, iterations=10, **kwargs
):
    """
    Run estimate_density_iterative_fp32 in spatial tiles with optional overlap padding.
    Returns a full (H, W, J) tensor.
    """

   


    F, _, H, W = Xflat.shape
    J = codeflat.shape[-1]

    result_full = np.zeros((H, W, J), dtype=np.float32)

    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            # Compute padded tile bounds
            i_start = max(i - overlap, 0)
            i_end   = min(i + tile_size + overlap, H)
            j_start = max(j - overlap, 0)
            j_end   = min(j + tile_size + overlap, W)

            # Extract padded tile from all frames
            Xtile = Xflat[:, :, i_start:i_end, j_start:j_end]

            try:
                et_tile, _ = bardensr.spot_calling.estimate_density_iterative(
                    Xtile, codeflat, iterations=iterations, **kwargs
                )
                et_tile = et_tile.squeeze(axis=0)  # Remove batch dimension
            except Exception as e:
                print(f"Tile ({i}:{i_end}, {j}:{j_end}) failed: {e}")
                continue

            # Determine where to crop the tile output
            crop_top    = i - i_start
            crop_bottom = crop_top + min(tile_size, H - i)
            crop_left   = j - j_start
            crop_right  = crop_left + min(tile_size, W - j)

            et_cropped = et_tile[crop_top:crop_bottom, crop_left:crop_right, :]

            result_full[i:i+et_cropped.shape[0], j:j+et_cropped.shape[1], :] = et_cropped

    # return np.squeeze(result_full, axis=0) if result_full.shape[0] == 1 else result_full
    return result_full[np.newaxis, ...]# if result_full.shape[0] == 1 else result_full

def call_genes_large_data(image, codeflat, codebook, genenames,thresh_refined, noisefloor, len_wid, round_num, fov_sample, find_thresh = False):

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    #image = image[round_num]
    image = np.stack(image, axis = 0)
    # image = image.astype(np.float32, copy=False)
    print(len(image))
    print(image[0].shape)
    Xflat = image.reshape(image.shape[0]*image.shape[1],1,image.shape[2], image.shape[3])
    # Xflat = image.reshape(image.shape[0]*image.shape[1],1,image.shape[2], image.shape[3]).astype(np.float32)
    # Xflat = Xflat.astype(np.float32)
    # codeflat = codeflat.astype(np.float32)

    print("Image dtype:", image.dtype)
    print("codeflat dtype:", codeflat.dtype)
    print("thresh_refined dtype:", np.asarray(thresh_refined).dtype)


    #  et=bardensr.spot_calling.estimate_density_singleshot(Xflat,codeflat,0.05)
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
    pixel_values = spots.apply(lambda row: et_squeezed[int(row['m2']), int(row['m1']), int(row['j'])], axis=1)    
    # Add the pixel value as a new column
    spots['pixel_val'] = pixel_values
    # spots['pixel_val'] = pd.Series(pixel_values.values, index=spots.index)
   
    # Compute the j index with the maximum value at each (m1, m2)
    max_j_array = np.argmax(et_squeezed, axis=2)  # shape: (2304, 2304)

    # Add a column to DataFrame indicating if that row is the max j at that (m1, m2)
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
        spots.to_csv(path + "output/fov"+str(fov_sample)+"_genes.csv")
    if find_thresh == True:
        return spots
    


def bardensr_registration(images, codeflat, x_dim, y_dim):
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----bardensr registration")

    R = images[0].shape[0]  # Number of rounds
    C = images[0].shape[1]  # Number of channels

    rounds = copy.deepcopy(images)

    for i in range(len(rounds)):  # for each FOV
        print(f"Registering FOV {i}...")

        # Get the stack for this FOV: shape (R, C, H, W)
        round_stack = rounds[i]

        # Run demons registration (new method)
        transforms, aligned_stack = register_stack_with_demons(round_stack)

        # Determine clipping shape
        xshape = min(aligned_stack.shape[-2], x_dim)
        yshape = min(aligned_stack.shape[-1], y_dim)

        # Save the aligned result back to the original container
        rounds[i][:, :, :xshape, :yshape] = aligned_stack[:, :, :xshape, :yshape]

    return rounds

## BARDENSR FULL IMAGE REGISTRATION
def split_image_into_patches(image, patch_size, overlap):
    patches = []
    image_height, image_width = image.shape[-2:]
    # Calculate the number of patches needed in rows and columns
    rows = ((image_height - patch_size[0]) // (patch_size[0] - overlap[0]))+1
    cols = ((image_width - patch_size[1]) // (patch_size[1] - overlap[1]))+1
    if image_height - (rows-1)*(patch_size[0] - overlap[0])-patch_size[0] >0:
        rows = rows+1
    if image_height - (cols-1)*(patch_size[0] - overlap[0])-patch_size[0] >0:
        cols = cols+1
    print(rows)
    for r in range(rows):
        for c in range(cols):
            # Calculate the start and end indices for the current patch
            start_r = max(0, r * (patch_size[0] - overlap[0]))
            end_r = min(start_r + patch_size[0], image_height)
            start_c = max(0, c * (patch_size[1] - overlap[1]))
            end_c = min(start_c + patch_size[1], image_width)
            # Extract patch from image
            patch = image[..., start_r:end_r, start_c:end_c]
            patches.append(patch)
    return patches

def find_position(original_image, larger_image, search_radius=10):
    # Calculate search area limits
    start_x = max(original_image.shape[1] // 2 - search_radius, 0)
    end_x = min(larger_image.shape[1] - original_image.shape[1] // 2 + search_radius, larger_image.shape[1])
    start_y = max(original_image.shape[0] // 2 - search_radius, 0)
    end_y = min(larger_image.shape[0] - original_image.shape[0] // 2 + search_radius, larger_image.shape[0])
    # Extract the search area from the larger image
    search_area = larger_image[start_y:end_y, start_x:end_x]
    # Calculate cross-correlation between original image and search area
    correlation = correlate2d(search_area, original_image, mode='same', boundary='fill', fillvalue=0)
    # Find the index of the maximum correlation value
    idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    # Calculate the start and end positions of the original image within the larger image
    start_x = start_x + idx[1] - original_image.shape[1] // 2
    end_x = start_x + original_image.shape[1]
    start_y = start_y + idx[0] - original_image.shape[0] // 2
    end_y = start_y + original_image.shape[0]
    return start_x, end_x, start_y, end_y

def register_patch(patch, codeflat,niter):
    # Perform registration on the patch (e.g., using BardenSR)
    # This is a placeholder function, replace it with your actual registration code
    #print(patch.shape)
    Xnorm = patch.reshape(patch.shape[0]*patch.shape[1], 1, patch.shape[2], patch.shape[3])
    with tf.device('/device:GPU:0'):
        print('start patch reg...')
        corrections = bardensr.registration.find_translations_using_model(Xnorm, codeflat, use_tqdm_notebook=False, niter=niter)
        flat, newt = bardensr.registration.apply_translations(Xnorm, corrections.corrections,mode='full')
        print('fin patch reg...')
    #print(flat.shape)
    start_x, end_x, start_y, end_y = find_position(patch[0,0], flat[0,0])
    # print(str(start_x) +" "+ str(end_x) +" "+ str(start_y) +" "+ str(end_y))
    flat = flat[:,:, :patch.shape[2], :patch.shape[3]]
    #print(flat.shape)
    #plt.imshow(flat[0,0])
    reflat = flat.reshape(patch.shape[0],patch.shape[1], patch.shape[2], patch.shape[3])
    return reflat  # Placeholder, replace with registered patch

def stitch_patches(patches, patch_size, image_shape, overlap):
    image = np.zeros(image_shape)
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"start bardensr registration")
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        ", len(patches), patches[0].shape)
    image_height, image_width = image.shape[-2:]
     # Calculate the number of patches needed in rows and columns
    rows = ((image_height - patch_size[0]) // (patch_size[0] - overlap[0]))+1
    cols = ((image_width - patch_size[1]) // (patch_size[1] - overlap[1]))+1
    if image_height - (rows-1)*(patch_size[0] - overlap[0])-patch_size[0] >0:
        rows = rows+1
    if image_height - (cols-1)*(patch_size[0] - overlap[0])-patch_size[0] >0:
        cols = cols+1
    idx=0
    for r in range(rows):
        for c in range(cols):
            # Calculate the start and end indices for the current patch
            start_r = max(0, r * (patch_size[0] - overlap[0]))
            end_r = min(start_r + patch_size[0], image_height)
            start_c = max(0, c * (patch_size[1] - overlap[1]))
            end_c = min(start_c + patch_size[1], image_width)
            # print(patches[idx].shape)
            # Extract patch from image
            image[..., start_r:end_r, start_c:end_c] = patches[idx]
            idx += 1
    return image

def register_fovs(images, codeflat, patch_size=(1000, 1000), overlap=(100, 100), niter = 50):
    num_fovs = len(images)
    registered_images = []
    for fov_idx in range(num_fovs):
        print("FOV: "+str(fov_idx))
        # Split the image into patches
        patches = split_image_into_patches(images[fov_idx], patch_size, overlap)
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"field of view: "+str(fov_idx))
        # Perform registration on each patch
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"registering "+str(len(patches))+" patches")
        registered_patches = [register_patch(patch, codeflat, niter) for patch in patches]
        # Stitch the registered patches back together
        registered_image = stitch_patches(registered_patches, patch_size, images[fov_idx].shape, overlap)
        # Store the registered FOV image
        registered_images.append(registered_image)
    # Stack the registered images along the first dimension to form the final registered image
    final_registered_image = np.stack(registered_images)
    return final_registered_image