import os
import argparse
import numpy as np
import pickle
import tensorflow as tf
import warnings

class Config(argparse.Namespace):
    """Default configuration for a N2V trainable CARE model.

    This class is meant to be used with :class:`N2V`.

    Parameters
    ----------
    kwargs : dict
             Overwrite (or add) configuration attributes (see below).

    Example
    -------
    >>> config = Config(filepath_homedir = "...",
                        filepath_rawdata = "...",
                        ...
                        )

    Attributes
    ----------


    """

    def __init__(self, **kwargs):
        
        self.filepath_homedir = "/home/manjari/Documents/starmap_pipeline/MA01_B2_TestPipeline/"

        self.filepath_rawdata = "/run/user/1006/gvfs/afp-volume:host=KebschullLab1.local,user=Dylan,volume=common_butterwort/manjari/MA04/manjari__e13_coronal_7_24_2023__MA04_A1_1000geneset_1nM_/"
        self.filepath_codebook = "/run/user/1006/gvfs/afp-volume:host=KebschullLab1.local,user=Dylan,volume=common_butterwort/manjari/MA04/manjari__e13_coronal_7_24_2023__MA04_A1_1000geneset_1nM_/Mouse_TFs_Prim_barcode_10_17_22 (2).csv"
        self.filepath_dapi = "/run/user/1006/gvfs/afp-volume:host=KebschullLab1.local,user=Dylan,volume=common_butterwort/manjari/MA04/manjari__e13_coronal_7_24_2023__MA04_A1_1000geneset_1nM_/round1/dapi.nd2"
        self.filepath_nissl = "/run/user/1006/gvfs/afp-volume:host=KebschullLab1.local,user=Dylan,volume=common_butterwort/manjari/MA04/manjari__e13_coronal_7_24_2023__MA04_A1_1000geneset_1nM_/round1/nissl.nd2"
        self.filepath_dapi_all_z = "//KebschullLab1//common_butterwort//manjari//MA27//1000gene_set_BSPEG_02_25_2024//Manjari_e13_coronal_02_25_2024_1000genes_BSPEG/round1/dapi_all_z.nd2"

        self.filepath_chalign = "starmap/align_channels.ijm"
        self.filepath_roalign = "starmap/align_rounds.ijm"
                
        self.round_index = [1, 2, 3, 4, 5]
        self.round_num = 5
        self.fov_num = 28

        # for channel & round alignment
        self.align_method = 'ImageJ' # option with 'ImageJ' or 'OpenCV'
        self.dapi_round = 1
        self.fov_align = 26
        self.round_align = 0
        # for color-bleed correction
        self.cc_coeff = [0.05, 0.45, 0.2, 0.05]
        self.fdr_thresh = 0.05

        # for normalizaiton
        self.fov_minmax = [33, 26, 21, 14, 9, 8, 15, 20, 27, 32, 31, 28, 19, 16, 7] # fovs counted in min-max normalization
        self.radius = 30 # the radius of rolling ball subtraction
        self.bardensr_patch_size=(1000, 1000), 
        self.bardensr_overlap=(100, 100)
        self.bardensr_niter = 50
        

        # gene calling
        self.base_code = ['G', 'T', 'C', 'A'] # order of base in codebook
        self.thresh_refined = 0.6 # threshold for gene calling
        self.noisefloor = 0.04
        self.spot_poolsize = (np.int64(0),np.int64(3),np.int64(3)) # spots size for gene calling  original (np.int64(0),np.int64(3),np.int64(3))
        self.len_wid = 500 # size of patches
        self.nogene_keyword = "no_gene"

        # cell segmantation
        self.cellpose_model =  'cyto3'#'cyto3'
        self.cellpose_channel = [0, 1] # original [0,1]
        self.cellpose_diameter = 35
        self.expand_pixel = 5

        # fov dimentions
        self.dim = 2304
        self.xdim = 2304
        self.ydim = 2304
        # self.overlap = 0.2
                
        self.ifov = 26
        self.igenes = ["Slc17a6", "Gad1", "Acan", "GFP", "nogene1"]

        self.SpGene = None # 'ChAlign', 'RoAlign', 'CbCorrection', 'Normalization', 'bardensrReg', 'GeneCalling', 'Finished'
        self.EpGene = None
        self.CpGene = {'ChAlign':True, 'RoAlign':True, 'CbCorrection':True, 'Normalization':True, 'BardensrReg': True, 'GeneCalling':True, 'Finished':True}
        self.SpMask = None # 'PosReg', 'CellSeg', 'ExpMask', 'Gene2Mask', 'RmvOverlap', 'Finished'
        self.EpMask = None
        self.CpMask = {'PosReg': True, 'CellSeg':True, 'ExpMask':True, 'Gene2Mask':True, 'RmvOverlap':True, 'Finished':True} # , '2AnnData':True
        self.grid_stitch = True # apply grid stitch to register FOVs when True
        self.bardensr_reg = True # run bardensr registration on normalized images(masks applied) when True
        self.find_param = False # find the threshold for gene calling when True
        self.cellpose_denoise = False # use denoising cellpose model when True
        self.remove_overlap = False # remove cells and gene calls in overlapped area when True

        for k in kwargs:
            setattr(self, k, kwargs[k])
            
        self.round_num = len(self.round_index)
        self.round_index0 = [i-1 for i in self.round_index]

        self.filepath_loadimg = self.filepath_homedir+"STARmap_loadImg/" # path to directory that saves loaded image files
        self.filepath_gridstch = self.filepath_loadimg+"stitch_images/" # path to directory that saves loaded image files
        self.filepath_imagej = self.filepath_homedir+"STARmap_ImageJ/" # path to directory that saves intermediate tiff files from ImageJ
        self.filepath_output = self.filepath_homedir+"STARmap_output/" # path to directory that saves useful output files
        self.filepath_val = self.filepath_output+"validation/" # path to directory that saves figures for validation
        self.filepath_alg = self.filepath_val+"alignment/" # path to directory that saves tiff files after alignment in validation
        
        self.GPU = tf.test.gpu_device_name()
        self._makeCP()
        self._build_directory()
        self.is_valid()

    def update_parameters(self, **kwargs): # , allow_new=True
        # if not allow_new:
        #     attr_new = []
        #     for k in kwargs:
        #         try:
        #             getattr(self, k)
        #         except AttributeError:
        #             attr_new.append(k)
        #     if len(attr_new) > 0:
        #         raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])

        self._makeCP()
        self.is_valid()
    
    def is_valid(self):
        # if self.filepath_dapi == "" and self.filepath_nissl == "" and self.filepath_dapi_all_z == "":
        #     raise ValueError("Missing cell information (dapi/nissl) file.")
        # if self.filepath_dapi != "" and self.filepath_nissl == "" and self.cellpose_channel != [0, 1]:
        #     raise ValueError("Incorrect corresponding cellpose channel.")
        # if self.filepath_dapi == "" and self.filepath_nissl != "" and self.cellpose_channel != [0, 1]:
        #     raise ValueError("Incorrect corresponding cellpose channel.")
        # if self.filepath_dapi != "" and self.filepath_nissl != "" and self.cellpose_channel != [2, 3]:
        #     raise ValueError("Incorrect corresponding cellpose channel.")
        # if self.filepath_dapi_all_z != "" and

        if self.GPU == "":
            warnings.warn("GPU is not detected")
            
        filename = self.filepath_val+'config.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)

    def _makeCP(self):
        if self.SpGene != None:
            for key in self.CpGene:
                if key == self.SpGene:
                    break
                self.CpGene[key] = False
        
        if self.EpGene != None:
            for key in reversed(self.CpGene):
                self.CpGene[key] = False
                if key == self.EpGene:
                    break

        if self.SpMask != None:
            for key in self.CpMask:
                if key == self.SpMask:
                    break
                self.CpMask[key] = False

        if self.EpMask != None:
            for key in reversed(self.CpMask):
                self.CpMask[key] = False
                if key == self.EpMask:
                    break

    def _build_directory(self):
        if not os.path.exists(self.filepath_homedir):
            os.makedirs(self.filepath_homedir)
        if not os.path.exists(self.filepath_loadimg):
            os.makedirs(self.filepath_loadimg)
        if not os.path.exists(self.filepath_gridstch):
            os.makedirs(self.filepath_gridstch)
        if not os.path.exists(self.filepath_imagej):
            os.makedirs(self.filepath_imagej)
        if not os.path.exists(self.filepath_output):
            os.makedirs(self.filepath_output)
        if not os.path.exists(self.filepath_val):
            os.makedirs(self.filepath_val)
        if not os.path.exists(self.filepath_alg):
            os.makedirs(self.filepath_alg)