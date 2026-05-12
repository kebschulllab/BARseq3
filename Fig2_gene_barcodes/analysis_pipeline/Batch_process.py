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
    "round_index": [1,2],
    "radius": 15,
    "fdr_thresh": 0.1,
    "expand_pixel": 7,
    "ifov": 0,
    "igenes": ["Slc17a6", "Gad1", "Atoh1", "Sox14"],
    "SpGene": 'ChAlign',# 'ChAlign', 'RoAlign', 'CbCorrection', 'Normalization', 'bardensrReg', 'GeneCalling', 'Finished'
    "EpGene": "Finished",#"Finished", # 'PosReg', 'CellSeg', 'ExpMask', 'Gene2Mask', 'RmvOverlap', 'Finished'
    "SpMask": "PosReg",
    "EpMask": "Finished",
    "grid_stitch": False,
    "bardensr_reg": False,
    "find_param": False,
    "align_method": 'OpenCV',
    "round_num": 2,
    "thresh_refined": 0.85,
    "bardensr_patch_size": (1000, 1000),
    "bardensr_overlap": (100, 100),
    "bardensr_niter": 10,
    "round_align": 1,
    "dapi_round": 1
}
'''

'''
# Define your list of sample-specific configurations
sample_subsets = [
    {
        "name": "Brain1",
        "filepath_homedir": r"Y:\Huihui\HH\STARmap01202026\Analysis_BARseq3_Transcriptome\Barcode_Gene\Brain1\\",
        "filepath_rawdata": r"Y:\Huihui\HH\STARmap01202026\Analysis_BARseq3_Transcriptome\Barcode_Gene\Brain1\\",
        "filepath_codebook": r"Y:\Huihui\HH\STARmap01202026\Analysis_BARseq3_Transcriptome\Barcode_Gene\Brain1\Gene_List.csv",
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
''' #Mouse1

    {

        "filepath_homedir": r"Y:\Huihui\BARseq2\BARseq2_ana\Gene_only\Brain1\\",
        "filepath_rawdata": r"Y:\Huihui\BARseq2\BARseq2_ana\Gene_only\Brain1\\",
        "filepath_codebook": r"Y:\Huihui\BARseq2\BARseq2_ana\Brain1\Barcode_List.csv",

        "filepath_homedir": r"Y:\Huihui\BARseq2\BARseq3_ana\Gene_only\Brain1\\",
        "filepath_rawdata": r"Y:\Huihui\BARseq2\BARseq3_ana\Gene_only\Brain1\\",
        "filepath_codebook": r"Y:\Huihui\BARseq2\BARseq3_ana\Brain1\Gene_List.csv",
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 9,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain2_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain2_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain3_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain3_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain4_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain4_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain5_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain5_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain6_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain6_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 9,
        "fov_align": 4,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain7_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain7_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain1_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain1_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain2_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain2_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain3_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain3_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain4_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain4_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain5_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain5_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain6_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain6_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain7_1/",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain7_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    }
'''
'''

    {
        "name": "Brain2_1",
        "filepath_homedir": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain1_1/",
        "filepath_rawdata": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain1_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 12,
        "fov_align": 7,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain2_1/",
        "filepath_rawdata": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain2_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 8,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain3_1/",
        "filepath_rawdata": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain3_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 8,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain4_1/",
        "filepath_rawdata": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain4_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 8,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain5_1/",
        "filepath_rawdata": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain5_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 9,
        "fov_align": 4,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain6_1/",
        "filepath_rawdata": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain6_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 8,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain2_1/",
        "filepath_rawdata": "/mnt3/Huihui/Mouse_Analysis_barcode/Mouse3/Brain2_1/",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 12,
        "fov_align": 4,
        "fov_minmax": [1]
    }
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 9,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain2_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain2_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 3,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain1_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain1_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain2_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain2_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain3_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain3_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain4_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain4_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain5_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain5_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain6_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain6_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    },
    {
        "name": "Brain2_1",
        "filepath_homedir": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain7_1\\",
        "filepath_rawdata": r"S:\Huihui\Mouse_Analysis_barcode\Mouse2\Brain7_1\\",
        "filepath_codebook": r"S:\Huihui\Mouse_Analysis_barcode\Mouse1\Brain1_1\Barcode_List.csv",
        "fov_num": 6,
        "fov_align": 5,
        "fov_minmax": [1]
    }
'''