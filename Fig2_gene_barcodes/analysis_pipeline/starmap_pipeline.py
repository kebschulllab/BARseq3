import os
import pandas as pd
from datetime import datetime
#from path import path

from starmap.config import Config
from starmap.pipeline import Pipeline
# from starmap import io as io
# from starmap import cell_segmentation as cellseg

if __name__ == '__main__':
    config = Config(filepath_homedir = "../analysis_output/",
                    filepath_rawdata = "//KebschullLab2/shareFlash/manjari/MA30/",
                    filepath_codebook = "//KebschullLab2/shareFlash/manjari/MA30/analysis/CAMs_TFs_barcodes_final_8nogene.csv",
                    filepath_dapi_all_z = "//KebschullLab2/shareFlash/manjari/MA30/round1_dapi_redo/dapi_all_z.nd2",
                    filepath_dapi = "//KebschullLab2/shareFlash/manjari/MA30/round1_dapi_redo/dapi.nd2",
                    filepath_nissl = "//KebschullLab2/shareFlash/manjari/MA30/round1_dapi_redo/dapi.nd2",
                    round_index = [1, 2, 3, 4, 5, 6, 7, 8,9],
                    fov_num = 49,
                    fov_align = 2,
                    fov_minmax = [33, 26, 21, 14],
                    cellpose_model = 'cyto',
                    cellpose_channel = [2, 3],
                    radius = 30,
                    fdr_thresh = 0.4,
                    expand_pixel = 17,
                    ifov = 23,
                    igenes = ["Slc17a6", "Gad1", "Atoh1", "Sox14", "nogene1", "nogene2"],
                    SpGene = 'ChAlign', # 'ChAlign', 'RoAlign', 'CbCorrection', 'Normalization', 'bardensrReg', 'GeneCalling', 'Finished'
                    # EpGene = "CbCorrection",
                    #SpMask = "Finished", # 'PosReg', 'CellSeg', 'ExpMask', 'Gene2Mask', 'RmvOverlap', 'Finished'
                    # EpMask = "ExpMask",
                    grid_stitch = False,
                    bardensr_reg = True,
                    find_param = True,
                    cellpose_denoise = True,
                    remove_overlap = True,
                    align_method = 'OpenCV',
                    round_num = 9,
                    thresh_refined = 0.799,#; fdrmean: 0.20054059775766364; diff:-0.00054059775766363
                    bardensr_patch_size = (1000, 1000), 
                    bardensr_overlap = (100, 100),
                    bardensr_niter = 10
                    )

    pipeline = Pipeline(config)

    pipeline.run()

    if config.CpMask['RmvOverlap'] and config.remove_overlap:
        if config.grid_stitch:
            os.system('python remove_overlap_mp.py ' + config.filepath_homedir + ' position_reg ' + str(config.dim))
        else:
            os.system('python remove_overlap_mp.py ' + config.filepath_homedir + ' position_corr ' + str(config.dim))
        # os.system('python remove_overlap_mp.py %s' % config.filepath_homedir)
    
    #pipeline.save()
