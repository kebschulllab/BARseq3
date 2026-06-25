import os
import pandas as pd
from datetime import datetime
#from path import path
from starmap.config import Config
from starmap.pipeline import Pipeline

# Base/default config parameters
base_config = {
    "filepath_dapi_all_z": "",
    "filepath_nissl": "",
    "filepath_dapi": "",
    "round_index": [1,2,3,4,5],
    "radius": 5,
    "fdr_thresh": 0.1,
    "expand_pixel": 0,
    "ifov": 0,
    "igenes": ["Slc17a6", "Gad1"],
    "SpGene": 'ChAlign',# 'ChAlign', 'RoAlign', 'CbCorrection', 'Normalization', 'bardensrReg', 'GeneCalling', 'Finished'
    "EpGene": "Finished",#"Finished", # 'PosReg', 'CellSeg', 'ExpMask', 'Gene2Mask', 'RmvOverlap', 'Finished'
    "SpMask": "PosReg",
    "EpMask": "Finished",
    "grid_stitch": False,
    "bardensr_reg": False,
    "find_param": False,
    "align_method": 'OpenCV',
    "round_num": 5,
    "thresh_refined": 0.7,
    "bardensr_patch_size": (1000, 1000),
    "bardensr_overlap": (100, 100),
    "bardensr_niter": 10,
    "round_align": 2,
    "dapi_round": 2
}
'''

'''
# Define your list of sample-specific configurations
sample_subsets = [
    {
        "name": "Brain1",
        "filepath_homedir": r"",
        "filepath_rawdata": r"",
        "filepath_codebook": r"",
        "fov_num": 5,
        "fov_align": 1,
        "fov_minmax": [1]
    }
    # Add more samples as needed...
]
if __name__ == '__main__': 
# Process all samples
    for subset in sample_subsets:
        print(f" Processing sample: {subset['name']}")

        # Merge base config with sample-specific config
        config_params = {**subset, **base_config}
        print(config_params)
        config = Config(**config_params)

        # Run pipeline
        pipeline = Pipeline(config)
        pipeline.run()

        # Run remove_overlap if required
        if config.CpMask.get('RmvOverlap') and getattr(config, 'remove_overlap', False):
            dim_arg = str(config.dim) if hasattr(config, 'dim') else ""
            if config.grid_stitch:
                os.system(f'python remove_overlap_mp.py {config.filepath_homedir} position_reg {dim_arg}')
            else:
                os.system(f'python remove_overlap_mp.py {config.filepath_homedir} position_corr {dim_arg}')

