import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["JAVA_HOME"] ="/home/manjari/Downloads/Fiji.app/java/linux-amd64/zulu8.60.0.21-ca-fx-jdk8.0.322-linux_x64/jre/lib/amd64/server/"
import platform
if platform.system() == 'Linux':
    os.environ["JAVA_HOME"] = "/home/manjari/Downloads/Fiji.app/java/linux-amd64/zulu8.60.0.21-ca-fx-jdk8.0.322-linux_x64/jre/lib/amd64/server/"
elif platform.system() == 'Windows':
    os.environ["JAVA_HOME"] = "/Users/Manjari/Fiji.app/java/win64/zulu8.60.0.21-ca-fx-jdk8.0.322-win_x64/jre/"
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL, IPython.display
import bardensr, bardensr.plotting
import copy, shutil
from datetime import datetime
import imagej
from cellpose import plot
import tifffile, argparse, warnings
warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from .config import Config
from . import io as io
from . import preprocessing as preproc
from . import gene_calling as genecall
from . import cell_segmentation as cellseg
from . import validation as val

class Pipeline(object):
    """
    The Pipeline class to run a starmap pipeline for spacial transcriptomics.

    Parameters
    ----------
    config : :class:`n2v.models.N2VConfig` or None
        Valid configuration of N2V network (see :func:`N2VConfig.is_valid`).
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).

    Example
    -------
    >>> pipeline = Pipeline(config)

    Attributes
    ----------
    config : :class:`n2v.models.N2VConfig`
        Configuration of pipeline.
    """

    def __init__(self, config, reload=False):
        self.config = config # ! need to be saved
        if self.config.SpGene != None or self.config.SpMask != None:
            reload = True
        self._build_directory()
        #self.ij = imagej.init('sc.fiji:fiji:2.5.0', mode='interactive')
        if not reload:
            self._load()
        else:
            self._reload()
        vars(self.config)

    def kill(self):
        del self

    def _build_directory(self):
        if not os.path.exists(self.config.filepath_homedir):
            os.makedirs(self.config.filepath_homedir)
        if not os.path.exists(self.config.filepath_loadimg):
            os.makedirs(self.config.filepath_loadimg)
        if not os.path.exists(self.config.filepath_gridstch):
            os.makedirs(self.config.filepath_gridstch)
        if not os.path.exists(self.config.filepath_imagej):
            os.makedirs(self.config.filepath_imagej)
        if not os.path.exists(self.config.filepath_output):
            os.makedirs(self.config.filepath_output)
        if not os.path.exists(self.config.filepath_val):
            os.makedirs(self.config.filepath_val)
        if not os.path.exists(self.config.filepath_alg):
            os.makedirs(self.config.filepath_alg)

    def _load(self):
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "run pipeline from fresh")
        self._load_images_spots() # self.image_spots, self.config.round_num, self.config.fov_num
        self._load_images_cells() # self.dapi_images, self.nissl_iamges, self.position
        self.codebook, self.genenames, self.codeflat = genecall.create_codebook(self.config.filepath_codebook,
                                                                                self.config.round_index0, self.config.base_code)

    def _reload(self):
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "restart the pipeline from some point")
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----reload dapi/nissl images")
        # self.dapi_images = np.load(self.config.filepath_loadimg+'dapi.npy')
        # self.dapi_all_z_images = np.load(self.config.filepath_loadimg+'dapi_all_z.npy')
        self._reload_images_cells()
        # load data for gene calling
        if self.config.SpGene != None:
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----reload intermediate data for gene calling")
            if self.config.SpGene != 'Finished':
                fileImage = {'ChAlign':'image_origin', 'RoAlign':'image_origin', 'CbCorrection':'image_cropped', 'Normalization':'image_corrected', 'BardensrReg':'image_preped', 'GeneCalling':'image_preped'}
                if self.config.bardensr_reg: fileImage['GeneCalling'] = 'image_registered'
                self.image_spots = io.open_hdf5(self.config.filepath_output + fileImage[self.config.SpGene] + '.hdf5')
                self.config.round_num, self.config.fov_num = len(self.image_spots[0]), len(self.image_spots)
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----    finish reloading image spots from: " + fileImage[self.config.SpGene] + '.hdf5')
                if not self.config.CpGene['RoAlign']: # os.path.isfile(self.config.filepath_output + 'image_masks.hdf5')
                    self.image_masks = io.open_hdf5(self.config.filepath_output + 'image_masks.hdf5')
                    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----    finish reloading image masks from: " + 'image_masks.hdf5')
            if self.config.SpGene == 'Finished':
                self.gene_called = pd.read_csv(self.config.filepath_output+'gene_called.csv')
            self.codebook, self.genenames, self.codeflat = genecall.create_codebook(self.config.filepath_codebook, # ! put intialize in pipeline function
                                                                                    self.config.round_index0, self.config.base_code)
        # load data for mask segmentation
        if self.config.SpMask != None:
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----reload intermediate data for cell segmentation")
            if self.config.CpMask['PosReg']:
                self.position = pd.read_csv(self.config.filepath_loadimg + 'position_org.csv')
            else:
                if self.config.grid_stitch:
                    self.position = pd.read_csv(self.config.filepath_loadimg + 'position_reg.csv')
                else:
                    self.position = pd.read_csv(self.config.filepath_loadimg + 'position_corr.csv')
            if self.config.SpMask not in ['Finished', 'PosReg', 'CellSeg']:
                fileMask = {'ExpMask':'mask_segmented', 'Gene2Mask':'mask_expanded', 'RmvOverlap':'mask_expanded'}
                self.cell_masks = io.open_hdf5(self.config.filepath_output + fileMask[self.config.SpMask] + '.hdf5')
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----    reload cell masks from: " + fileMask[self.config.SpMask] + '.hdf5')
                fileGene = {'CellSeg':'gene_called', 'ExpMask':'gene_called', 'Gene2Mask':'gene_called', 'RmvOverlap':'gene_mapped'}
                self.gene_called = pd.read_csv(self.config.filepath_output + fileGene[self.config.SpMask] + '.csv')
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----    reload gene calls from: " + fileGene[self.config.SpMask] + '.csv')

        # for gene_file in ['gene_trimmed', 'gene_mapped', 'gene_called']:
        #     if os.path.isfile(self.config.filepath_output + gene_file + '.csv'):
        #         self.gene_called = pd.read_csv(self.config.filepath_output + gene_file + '.csv')
        #         print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----reload gene calls from: " + gene_file + '.csv')
        #         break

    def run(self):
        if self.config.CpGene['ChAlign'] or self.config.CpGene['RoAlign']:
            self.image_registration() # relative pathway like "../" is not working
        if self.config.CpGene['CbCorrection']: self.colorbleed_correction()
        if self.config.CpGene['Normalization']: self.background_subtraction() # ! too much resize
        if self.config.CpGene['BardensrReg'] and self.config.bardensr_reg: self.bardensr_registration()
        if self.config.CpMask['PosReg']: self.position_registration()
        if self.config.CpGene['GeneCalling']: self.gene_calling()
        if self.config.SpMask != 'Finished': self.cell_segmentation()

    def position_registration(self):
        # generally correcte the position  --------------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----generally correcte the position")
        self.position = pd.read_csv(self.config.filepath_loadimg + 'position_org.csv')
        self.position = preproc.align_fovs(self.position, self.config.dim) # img_list is dapi_image added with zero fovs
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "    ----save corrected positions")
        self.position.to_csv(self.config.filepath_loadimg + 'position_corr.csv', index=False)

        if self.config.grid_stitch:
            # register position with grid stitching --------------------------------------------------
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----register position with grid stitching")
            self.position = preproc.grid_stitch(self.ij, self.dapi_images, self.position,
                                                self.config.xdim, self.config.ydim, self.config.filepath_gridstch)
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "    ----save registered positions")
            self.position.to_csv(self.config.filepath_loadimg + 'position_reg.csv', index=False)
            shutil.copyfile(self.config.filepath_gridstch+"img_t1_z1_c1",
                            self.config.filepath_val+"after_grid_stitching.tiff")

    def image_registration(self):
        # alignment(channel, round) with ImageJ -----------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----alignment(channel, round) with ImageJ")
        if self.config.align_method == 'ImageJ':
            if self.config.CpGene['ChAlign']:
                io.write_registration_file(self.image_spots, self.config.round_num, self.config.fov_num, self.config.filepath_imagej)
            if self.config.CpGene['ChAlign']: preproc.channel_alignment(self.ij, self.config.round_num, self.config.fov_align, self.config.filepath_imagej, self.config.filepath_chalign)
            if self.config.CpGene['RoAlign']: preproc.round_alignment(self.ij, self.config.round_num, self.config.round_align, self.config.filepath_imagej, self.config.filepath_roalign)
            if self.config.CpGene['ChAlign'] or self.config.CpGene['RoAlign']:
                self.image_spots = io.read_registration_file((self.config.fov_num, self.config.round_num, 4, self.config.xdim, self.config.ydim),
                                                            self.config.round_align, self.config.filepath_imagej)
            # self.image_spots = preproc.image_alignment(self.ij, self.image_spots, self.config.fov_align,
            #                                            self.config.CpGene['ChAlign'], self.config.CpGene['RoAlign'],
            #                                            self.config.filepath_imagej, self.config.filepath_chalign, self.config.filepath_roalign)
        elif self.config.align_method == 'OpenCV':
            if self.config.CpGene['ChAlign']: 
                self.image_spots = preproc.channel_alignment_opencv(self.image_spots, self.config.fov_align) ### COMMENT OUT TO NOT USE CHANNEL ALIGNEMNT
                #self.image_spots = preproc.shading_correction(self.image_spots, "shading_corrimg_2304.tiff")
                imgs = [bardensr.plotting.lutup(*x,sc=.5,normstyle='each') for x in self.image_spots[self.config.ifov][:,:,500:1000, 500:1000]]
                tifffile.imsave(self.config.filepath_val+'registration_chalign_QC_fov'+str(self.config.ifov)+'.tif', np.array(imgs)) # could be opened with IFJI or napari
                imgs = [PIL.Image.fromarray(x) for x in imgs]
                imgs[0].save(self.config.filepath_val+'registration_chalign_QC_fov'+str(self.config.ifov)+'.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0) # 


                imgs = [bardensr.plotting.lutup(*x,sc=.5,normstyle='each') for x in self.image_spots[self.config.ifov][:,:,0:500, 0:500]]
                tifffile.imsave(self.config.filepath_val+'registration_chalign_QC_2_fov'+str(self.config.ifov)+'.tif', np.array(imgs)) # could be opened with IFJI or napari
                imgs = [PIL.Image.fromarray(x) for x in imgs]
                imgs[0].save(self.config.filepath_val+'registration_chalign_QC_2_fov'+str(self.config.ifov)+'.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0) # 

                imgs = [bardensr.plotting.lutup(*x,sc=.5,normstyle='each') for x in self.image_spots[self.config.ifov][:,:]]
                tifffile.imsave(self.config.filepath_val+'registration_chalign_QC_full_fov'+str(self.config.ifov)+'.tif', np.array(imgs)) # could be opened with IFJI or napari
                imgs = [PIL.Image.fromarray(x) for x in imgs]
                imgs[0].save(self.config.filepath_val+'registration_chalign_QC_full_fov'+str(self.config.ifov)+'.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0) # 


            if self.config.CpGene['RoAlign']: 
                self.image_spots = preproc.round_alignment_opencv(self.image_spots, self.config.round_index0, self.config.filepath_val, self.config.round_align) # ! need to add round_align & dapi
            
        io.save_hdf5(self.image_spots, self.config.filepath_output+'image_aligned.hdf5')

        # VAL: save .gif for registration QC
        imgs = [bardensr.plotting.lutup(*x,sc=.5,normstyle='each') for x in self.image_spots[self.config.ifov][:,:,:,:]]
        tifffile.imsave(self.config.filepath_val+'registration_QC_fov'+str(self.config.ifov)+'.tif', np.array(imgs)) # could be opened with IFJI or napari
        imgs = [PIL.Image.fromarray(x) for x in imgs]
        imgs[0].save(self.config.filepath_val+'registration_QC_fov'+str(self.config.ifov)+'.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=1000, loop=0) # 

        # # VAL: copy files after alignment ---------------------------------
        # print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----plot tiff files after alignment")
        # images = []
        # for i in self.config.round_index: 
        #     shutil.copyfile(self.config.filepath_imagej+"round"+str(i)+"/data_chalign_r"+str(i)+"_fov"+str(self.config.ifov)+"_chcorr.tiff",
        #                     self.config.filepath_alg+"data_chalign_r"+str(i)+"_fov"+str(self.config.ifov)+"_chcorr.tiff")
        #     img = tifffile.imread(self.config.filepath_imagej+"round"+str(i)+"/data_chalign_r"+str(i)+"_fov"+str(self.config.ifov)+"_roundcorr.tiff")
        #     images.append(np.max(img, axis=0))
        # merged_image = np.dstack(images)
        # tifffile.imwrite(self.config.filepath_alg+"data_chalign_fov"+str(self.config.ifov)+"_roundcorr.tiff", merged_image)

        if self.config.CpGene['RoAlign']:
            # crop images to get rid of alignment offsets -----------------------------
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----crop images to get rid of alignment offsets")
            self.image_spots, self.image_masks = preproc.crop_images(self.image_spots, self.config.xdim, self.config.ydim)
            io.save_hdf5(self.image_spots, self.config.filepath_output+'image_cropped.hdf5')
            io.save_hdf5(self.image_masks, self.config.filepath_output+'image_masks.hdf5')

    def colorbleed_correction(self):
        # colorbleed correction ---------------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----colorbleed correction")
        image_cropped = self.image_spots
        self.image_spots = preproc.colorbleed_correction_inversemat(self.image_spots)
        for i in range(len(self.image_spots)): # clip images to remove outlier pixel values
            self.image_spots[i] = np.clip(self.image_spots[i],10,8000)
        io.save_hdf5(self.image_spots, self.config.filepath_output+'image_corrected.hdf5')
        image_corrected = self.image_spots

        # VAL: plot color-bleed correction ----------------------------------------
        # image_cropped = io.open_hdf5_NxRxC(self.config.filepath_output+'image_cropped.hdf5', self.config.fov_num, self.config.round_num)
        # image_corrected = io.open_hdf5_NxRxC(self.config.filepath_output+'image_corrected.hdf5', self.config.fov_num, self.config.round_num)
        ifov = self.config.ifov
        plt.figure(figsize=(20,16))
        plt.subplot(2,2,1)
        bardensr.preprocessing.colorbleed_plot(np.clip(image_cropped[ifov][0,0],0,5000),
                                               np.clip(image_cropped[ifov][0,1],0,5000))
        plt.xlabel(f'color {0}')
        plt.ylabel(f'color {1}')
        plt.title('Before Colorbleed Correction: color-0 & color-1')
        plt.subplot(2,2,2)
        bardensr.preprocessing.colorbleed_plot(np.clip(image_cropped[ifov][0,2],0,5000),
                                               np.clip(image_cropped[ifov][0,3],0,5000))
        plt.xlabel(f'color {2}')
        plt.ylabel(f'color {3}')
        plt.title('Before Colorbleed Correction: color-2 & color-3')
        plt.subplot(2,2,3)
        bardensr.preprocessing.colorbleed_plot(np.clip(image_corrected[ifov][0,0],0,5000),
                                               np.clip(image_corrected[ifov][0,1],0,5000))
        plt.xlabel(f'color {0}')
        plt.ylabel(f'color {1}')
        plt.title('After Colorbleed Correction: color-0 & color-1')
        plt.subplot(2,2,4)
        bardensr.preprocessing.colorbleed_plot(np.clip(image_corrected[ifov][0,2],0,5000),
                                               np.clip(image_corrected[ifov][0,3],0,5000))
        plt.xlabel(f'color {2}')
        plt.ylabel(f'color {3}')
        plt.title('After Colorbleed Correction: color-2 & color-3')
        plt.savefig(self.config.filepath_val+'colorbleed_correction_fov'+str(self.config.ifov)+'.png')

    def background_subtraction(self):
        # normalization ----------------------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----normalization")
        with tf.device(self.config.GPU):
            self.image_spots = preproc.round_normalization(self.image_spots, self.config.fov_minmax, self.config.radius)
        print("Saving round normalization matrix....")
        io.save_hdf5(self.image_spots, self.config.filepath_output+'image_normed.hdf5')

        # apply cropped image_masks ----------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----apply cropped image_masks")
        image_preped = []
        for i in range(len(self.image_spots)):
            image_preped.append(preproc.apply_mask_to_images(self.image_spots[i], self.image_masks[i]))
        io.save_hdf5(image_preped, self.config.filepath_output+'image_preped.hdf5')
        self.image_spots = image_preped

    def bardensr_registration(self):
        with tf.device(self.config.GPU):
            self.image_spots = preproc.bardensr_registration(self.image_spots, self.codeflat,
                                                          self.config.xdim, self.config.ydim)
        
            #self.image_spots = preproc.register_fovs(self.image_spots, self.codeflat, self.config.bardensr_patch_size, self.config.bardensr_overlap)
        io.save_hdf5(self.image_spots, self.config.filepath_output+'image_registered.hdf5')

    def gene_calling(self):
        # call the genes & add global positions --------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----call the genes & add global positions")
        ### self.image_spots: image_preped [fovs, (round, channel, xdim, ydim)]
        with tf.device(self.config.GPU):
            if self.config.find_param:
                self.config.thresh_refined, self.config.noisefloor = genecall.find_params_gradient(self.image_spots, self.codeflat, self.genenames,
                                                                                                   self.config.nogene_keyword, self.config.filepath_val, self.config.fov_minmax, self.config.round_index0, self.config.fdr_thresh)
            self.gene_called = genecall.call_genes_large_data(self.image_spots, self.position, self.codeflat, self.codebook, self.genenames, 
                                                            self.config.thresh_refined, self.config.noisefloor, self.config.len_wid, self.config.round_index0)
            
            #self.gene_called = preproc.call_genes_large_data(self.image_spots, self.codeflat, self.codebook, self.genenames, 
             #                                               self.config.thresh_refined, self.config.noisefloor, self.config.len_wid, self.config.round_index0, fov_sample, find_thresh = False)
        self.gene_called.to_csv(self.config.filepath_output+'gene_called.csv', index=False)

        # VAL: check gene distribution ------------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----check gene distribution")
        num_genes=[]
        for i in np.array(np.unique(self.gene_called['Names'])):
            num_genes.append(len(self.gene_called[self.gene_called['Names'] == i]))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"total genes: " + str(np.sum(num_genes)))
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "        "+"total unique genes " + str(len(num_genes)))
        plt.figure()
        plt.bar(np.array(np.unique(self.gene_called['Names'])),num_genes)
        plt.xticks(np.array(np.unique(self.gene_called['Names'])), rotation='vertical')
        plt.title('Gene Distribution')
        plt.xlabel('Genes')
        plt.ylabel('Count')
        plt.savefig(self.config.filepath_val+'gene_results.png')
        
    def cell_segmentation(self):
        # Cell Segmentation ================================================================================================================================
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Start Cell Segmentation")

        if self.config.CpMask['CellSeg']:
            # find cell masks --------------------------------------------------
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----find cell masks")
            if self.config.cellpose_denoise:
                self.cell_masks = cellseg.denoise_segment_images(self.dapi_all_z_images)
            else:
                if (self.config.dapi_round != self.config.round_align) and self.dapi_images != None:
                    io.write_dapi_registration_file(self.dapi_images, self.config.fov_num, self.config.filepath_imageJ, 'dapi/')
                    preproc.dapi_alignment(self.ij, self.config.fov_num, self.config.dapi_round,
                                           self.config.filepath_imageJ, 'dapi/')
                    self.dapi_images = io.read_dapi_registration_file((self.config.fov_num, self.config.xdim, self.config.ydim),
                                                                      self.config.filepath_imageJ, 'dapi/')
                if (self.config.dapi_round != self.config.round_align) and self.nissl_images != None:
                    io.write_dapi_registration_file(self.nissl_images, self.config.fov_num, self.config.filepath_imageJ, 'nissl/')
                    preproc.dapi_alignment(self.ij, self.config.fov_num, self.config.dapi_round,
                                           self.config.filepath_imageJ, 'nissl/')
                    self.nissl_images = io.read_dapi_registration_file((self.config.fov_num, self.config.xdim, self.config.ydim),
                                                                      self.config.filepath_imageJ, 'nissl/')
                self.cell_masks = cellseg.find_masks(self.dapi_images, self.nissl_images,
                                                    self.config.cellpose_model, self.config.cellpose_channel, self.config.cellpose_diameter)
            io.save_hdf5(self.cell_masks, self.config.filepath_output+'mask_segmented.hdf5')

        if self.config.CpMask['ExpMask']:
            # expand cell masks ------------------------------------------------
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----expand cell masks")
            self.cell_masks = cellseg.expand_masks(self.cell_masks, self.config.expand_pixel)
            io.save_hdf5(self.cell_masks, self.config.filepath_output+'mask_expanded.hdf5')

        if self.config.CpMask['Gene2Mask']:
            # map gene spots to cells -------------------------------------------
            print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----map gene spots to cells")
            self.gene_called = cellseg.map_gene2cell(self.gene_called, self.cell_masks, self.position,
                                                     self.config.dim, self.config.xdim, self.config.ydim)
            self.gene_called.to_csv(self.config.filepath_output+'gene_mapped.csv', index=False)
            # io.save_hdf5(self.cell_masks, self.config.filepath_output+'mask_mapped.hdf5')

        # if self.config.CpMask['RmvOverlap'] and self.config.remove_overlap:
        #     # remove repeated gene spots within overlap area ----------------------
        #     print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----remove repeated gene spots within overlap area")
        #     self.gene_called, self.cell_masks, cell_deleted = cellseg.overlap_rmv_mp(self.gene_called, self.cell_masks, self.position)
        #     self.gene_called.to_csv(self.config.filepath_output+'gene_trimmed.csv', index=False)
        #     io.save_hdf5(self.cell_masks, self.config.filepath_output+'mask_trimmed.hdf5')
        #     cell_deleted = pd.DataFrame(cell_deleted)
        #     cell_deleted.to_csv(self.config.filepath_output+'cell_deleted.csv', index=False)
    
    def validation(self):
        val.plot_dots_brightness(self.config)
        val.plot_incorrect_genes(self.config)
        val.plot_fdr(self.config)
        val.plot_gene_per_cell(self.config)

    def save(self):
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Start Saving Output")
        if (self.config.CpMask['RmvOverlap'] and self.config.remove_overlap):
            self.gene_called = pd.read_csv(self.config.filepath_output + 'gene_trimmed.csv')
            self.cell_masks = io.open_hdf5(self.config.filepath_output + 'mask_trimmed.hdf5')
        else:
            self.gene_called = pd.read_csv(self.config.filepath_output + 'gene_mapped.csv')
            self.cell_masks = io.open_hdf5(self.config.filepath_output + 'mask_segmented.hdf5')

        # convert to annData ------------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----convert to annData")
        gene_anndata = cellseg.convert_to_annData(self.gene_called)
        gene_anndata.write_h5ad(self.config.filepath_output + 'gene_anndata.h5ad')
        # save cell boundaries
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----save cell boundaries")
        io.save_boundaries(self.cell_masks, self.position, self.config)
    
    def _load_images_spots(self): # ! function to load inter images
        # Loading Images ==============================================================
        # save max_proj images in rounds
        # print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----save max_proj images in rounds")
        # for i in self.config.round_index:
        #     np.save(self.config.filepath_loadimg + 'round'+str(i)+'.npy',
        #             io.read_gtca_images(self.config.filepath_rawdata + 'round'+str(i)+'/max_proj.nd2'))

        # read in round files ----------------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----read in round files")
        image_readin = []
        for i in self.config.round_index:
            image_readin.append(io.read_gtca_images(self.config.filepath_rawdata + 'round'+str(i)+'/max_proj.nd2'))
        self.image_spots = np.stack(image_readin) # shape: (R, FOV, C, X, Y)
        self.image_spots = np.swapaxes(self.image_spots,0,1) # shape: (FOV, R, C, X, Y)
        io.save_hdf5(self.image_spots, self.config.filepath_output+'image_origin.hdf5')
        self.config.round_num, self.config.fov_num = len(self.image_spots[0]), len(self.image_spots) # there should be at least one round
        print("ROUND NUM: "+str(self.config.round_num))
        # return image_rawnp, round_num, fov_num

    def _load_images_cells(self):
        # save dapi images & positions -------------------------------------------
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----save dapi images & positions")
        if self.config.filepath_dapi != "":
            [self.dapi_images, position_dapi] = io.read_dapi_images(self.config.filepath_dapi)
            np.save(self.config.filepath_loadimg+'dapi.npy', self.dapi_images)
            self.position = position_dapi
            self.position.to_csv(self.config.filepath_loadimg + 'position_org.csv', index=False)

        if self.config.filepath_nissl != "":
            [self.nissl_images, position_nissl] = io.read_dapi_images(self.config.filepath_nissl)
            np.save(self.config.filepath_loadimg+'nissl.npy', self.nissl_images)
            self.position = position_nissl
            # self.position.to_csv(self.config.filepath_loadimg + 'position_org.csv', index=False)
        
        if self.config.filepath_dapi_all_z != "" and self.config.cellpose_denoise:
            [self.dapi_all_z_images, position_dapi_all_z] = io.read_dapi_all_z_images(self.config.filepath_dapi_all_z)
            np.save(self.config.filepath_loadimg+'dapi_all_z.npy', self.dapi_all_z_images)
        # self.position = position_dapi_all_z
        # self.position.to_csv(self.config.filepath_loadimg + 'position_org.csv', index=False) 
        # return dapi_images, nissl_images, position_org

    def _reload_images_cells(self):
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----reload dapi/nissl images")
        if os.path.isfile(self.config.filepath_loadimg+'dapi.npy'):
            self.dapi_images = np.load(self.config.filepath_loadimg+'dapi.npy')
        else:
            self.dapi_images = None

        if os.path.isfile(self.config.filepath_loadimg+'nissl.npy'):
            self.nissl_images = np.load(self.config.filepath_loadimg+'nissl.npy')
        else:
            self.nissl_images = None
        
        if os.path.isfile(self.config.filepath_loadimg+'dapi_all_z.npy'):
            self.dapi_all_z_images = np.load(self.config.filepath_loadimg+'dapi_all_z.npy')
        else:
            self.dapi_all_z_images = None