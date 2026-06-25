import os
import pandas as pd
from datetime import datetime
from os.path import abspath
# print(abspath('../example_dataset/'))

from starmap.config import Config
from starmap.pipeline import Pipeline
# from starmap import io as io
# from starmap import cell_segmentation as cellseg

if __name__ == '__main__':
    config = Config(filepath_homedir = abspath("../example_dataset/"),
                    filepath_rawdata = "",
                    filepath_codebook = abspath('../example_dataset/')+"Acan_classB_11geneset_GFP_barcodes.csv",
                    filepath_dapi_all_z = "",
                    filepath_dapi = "",
                    filepath_nissl = "",
                    round_index = [1, 2, 3, 4],
                    fov_num = 3,
                    fov_align = 1,
                    fov_minmax = [0, 1, 2],
                    cellpose_model = 'cyto',
                    cellpose_channel = [0, 1],
                    radius = 10,
                    expand_pixel = 15,
                    ifov = 1,
                    igenes = ["Slc17a6", "Gad1", "Acan", "GFP", "nogene1", "nogene2"],
                    SpGene = "ChAlign", # 'ChAlign', 'RoAlign', 'CbCorrection', 'Normalization', 'bardensrReg', 'GeneCalling', 'Finished'
                    # SpMask = "PosReg", # 'PosReg', 'CellSeg', 'ExpMask', 'Gene2Mask', 'RmvOverlap', 'Finished'
                    grid_stitch = True,
                    bardensr_reg = True,
                    find_param = True,
                    cellpose_denoise = False,
                    remove_overlap = True
                    )

    pipeline = Pipeline(config)

    pipeline.run()

    if config.CpMask['RmvOverlap'] and config.remove_overlap:
        if config.grid_stitch:
            os.system('python remove_overlap_mp.py ' + config.filepath_homedir + ' position_reg ' + str(config.dim))
        else:
            os.system('python remove_overlap_mp.py ' + config.filepath_homedir + ' position_corr ' + str(config.dim))
        # os.system('python remove_overlap_mp.py %s' % config.filepath_homedir)
    
    pipeline.save()