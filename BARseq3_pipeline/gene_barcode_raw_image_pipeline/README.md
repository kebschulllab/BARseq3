# Get Started - Running Pipeline on Local Computer

## Installation (Less than 1h)



```bash
git clone --branch main --filter=blob:none --sparse https://github.com/kebschulllab/BARseq3.git

cd BARseq3

git sparse-checkout set BARseq3_pipeline/gene_barcode_raw_image_pipeline

cd BARseq3_pipeline/gene_barcode_raw_image_pipeline

```

### For both Linux and Windows


- use environment *starmap*: <code>conda env create -f environment.yml -n starmap</code>
- activate the environment: <code>conda activate BARseq3</code>
- make sure CUDA, CUDNN, and NVIDIA/GPU Drivers are properly installed
- install tensorflow based on the system: https://www.tensorflow.org/install/pip (you may need to downgrade Numpy version)
- install bardensr dependency: <code>pip install --upgrade git+https://github.com/jacksonloper/bardensr.git</code>
- **note: you must have CUDA properly downloaded and enabled for fast GPU processing
- install additional dependencies: <code>pip install n2v scanpy anndata cellpose pims-nd2 ipython imagej napari</code>

## Run pipeline locally (20-30 min)

- navigate to the "analysis_pipeline" folder
- modify configurations in the starmap_pipeline.py file and save
  
  - `filepath_rawdata`

  - `filepath_codebook`

  - `filepath_dapi`

  - `round_index = [1, 2, 3, 4, 5]`

  - `radius = 5`

  - `expand_pixel = 0`

  - `find_param = False`

  - `round_num = 5`

  - `thresh_refined = 0.7`

  - `round_align = 2`

  - `dapi_align = 2`
    
  - `open Batch_process.py in VS Code and run`
 
## For barcode analysis

  - `open Analysis_pipeline_for_barcodes.ipynb`
    
  - `modify file path and name`
    
  - `run each cell`

  -  `output (Excel file with genes and barcodes information https://zenodo.org/records/19024678/files/BARseq3_Transcriptome_Barcode.tar?download=1)`


## Folder structure


```text
brain/
├── STARmap_ImageJ/
│   ├── round1/
│   ├── round2/
│   ├── round3/
│   ├── round4/
│   └── round5/
├── STARmap_loading/
│   ├── nissl.npy
│   ├── position_corr.csv
│   └── position_org.csv
└── STARmap_output/
```
## Code structure

└── analysis_pipeline  
&emsp;&emsp;├── starmap_pipeline.py              *debug version  
&emsp;&emsp;├── config/                          *save config files  
&emsp;&emsp;│&emsp;&emsp;└── starmap_config_experiment.json  
&emsp;&emsp;├── log/                             *save output log  
&emsp;&emsp;│&emsp;&emsp;└── output_example_mm_dd.txt  
&emsp;&emsp;└── starmap/                         *save starmap scripts  
&emsp;&emsp; &emsp;&emsp;├── io.py  
&emsp;&emsp; &emsp;&emsp;├── preprocessing.py  
&emsp;&emsp; &emsp;&emsp;├── gene_calling.py  
&emsp;&emsp; &emsp;&emsp;├── cell_segmentation.py  
&emsp;&emsp; &emsp;&emsp;├── stitch_images.ijm  
&emsp;&emsp; &emsp;&emsp;├── align_channels.ijm  
&emsp;&emsp; &emsp;&emsp;└── align_rounds.ijm  

## Raw Data Structure
please make sure that your raw data folder will structured like the following with max-projected file named "max_proj.nd2" and saved in each directory for roundx, or you can change the code part for how raw data is read in.

└── filepath_rawdata/  
&emsp;&emsp;├── round1/  
&emsp;&emsp;│&emsp;&emsp;├── ChannelG,T,C,A_Seq0000.nd2  
&emsp;&emsp;│&emsp;&emsp;└── max_proj.nd2  
&emsp;&emsp;├── round2/  
&emsp;&emsp;│&emsp;&emsp;├── ChannelG,T,C,A_Seq0000.nd2  
&emsp;&emsp;│&emsp;&emsp;└── max_proj.nd2  
&emsp;&emsp;└── .../  

## Notes for config
|variables           |notes                                          |
| ------------------ | --------------------------------------------- |
|**filepath_rawdata**|path to directory that saves raw max_proj files|
|**filepath_homedir**|path to output home directory|
|**filepath_codebook**|path to codebook (.csv file)|
|**filepath_dapi**|path to dapi img|
|**round_index**|total rounds [1 ,2 ,3, 4, 5, 6] start from 1 :exclamation: should include more than one round|
|**fov_align**|change to field of view with as many dots as possible|
|cc_coeff|[zero_one, one_zero, two_three, three_two]|
|**fov_minmax**|specify the fovs for calculating the minimum and maximum values|
|radius|set the radius of background subtraction (larger -> less background subtracted)|
|base_code|the list of base in codebook|
|**find_param**|1 call find_params() to calculate thresh_refined, noisefloor|
|thresh_refined, noisefloor|for gene calling|
|expand_pixel|how many pixels to expand by|
|**ifov**|specify the number of fov interested|
|**igenes**|specify the number of genes interested|

## About output results
you can specify the filepath of the directory(*filepath_homedir*) which stores all the output  
all the output files will be stored in *filepath_homedir* like:  

└── filepath_homedir/  
&emsp;&emsp;├── STARmap_ImageJ/  
&emsp;&emsp;│ &emsp;&emsp;├── round1/  
&emsp;&emsp;│ &emsp;&emsp;│ &emsp;├── transf  
&emsp;&emsp;│ &emsp;&emsp;│ &emsp;└── .tiff files (chcorr & roundcorr)  
&emsp;&emsp;│ &emsp;&emsp;└── .../  
&emsp;&emsp;├── STARmap_loadImg/  
&emsp;&emsp;│ &emsp;&emsp;├── roundx.npy...  
&emsp;&emsp;│ &emsp;&emsp;├── dapi.npy  
&emsp;&emsp;│ &emsp;&emsp;├── position_org.csv  
&emsp;&emsp;│ &emsp;&emsp;├── position_corr.csv  
&emsp;&emsp;│ &emsp;&emsp;└── position_reg.csv  
&emsp;&emsp;└── STARmap_output/  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── validation/  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;├── alignment/  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;├── *.tiff* after channel alignment  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;│&emsp;&emsp;└── *.tiff* after round alignment  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;├── thresh_refined.txt  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;├── starmap_config.json  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;├── full_dapi_img.png  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;├── gene_results.png  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;├── gene_all.png  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;│&emsp;&emsp;└── gene_igenes.png  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── image_aligned.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── image_cropped.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── image_masks.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── image_corrected.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── image_normed.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── image_preped.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── gene_called.csv  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── gene_mapped.csv  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── gene_trimmed.csv  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── gene_anndata *file type not decided yet  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── mask_expanded.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── mask_mapped.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;├── mask_trimmed.hdf5  
&emsp;&emsp;&nbsp;&nbsp;&emsp;&emsp;└── cell_deleted.txt  

### Notes for output  
1. STARmap_ImageJ/: saves all the intermediate files while performing alignment with PyImageJ  
   - round*x*: files from each round x
     - transf/: saves both direct and reverse transformation configs for channel alignment and round alignment for each round
2. STARmap_loadImg/: saves all the files processed from raw image data  
   - round*x*.npy: raw data for each round x
   - dapi.npy: raw data for dapi image
   - positions_org.csv: original positions of fovs loading from dapi image
   - positions_corr.csv: positions corrected by align all fovs
   - **positions_reg.csv**: positions registered through grid stitching
3. STARmap_output/: saves intermediate data from each subprocessing
   - validation/: saves figs for validation
     - alignment/: saves tiff files after channel and round alignment in ifov
     - thresh_refined.txt: saves the thresh_refined and noisefloor if find_params is TRUE
     - starmap_config.json: back up the json config
     - **full_dapi_img.png**: plot the full dapi image
     - **gene_results**.png: a bar plot summarizes gene calls
     - **gene_all.png**: plot all the gene calls distribution over the dapi image
     - **gene_igenes.png**: plot the gene that is interested in over the dapi image
   - image_aligned.hdf5: data of round images after alignment
   - image_cropped.hdf5: data of round images after croping masks
   - image_masks.hdf5: data of masks after croping masks
   - image_corrected.hdf5: data of round images after color correction
   - image_normed.hdf5: data of round images after normalization
   - image_aligned.hdf5: data of round images after alignment
   - **image_preped.hdf5**: data of round images after applying masks
   - gene_called.csv: result of genes calls
   - gene_mapped.csv: result of genes calls mapped to cell masks
   - **gene_trimmed.csv**: result of genes calls trimmed through removing repeated cells within overlap
   - gene_anndata: result of genes calls in annData *file type not decided yet
   - mask_expanded.hdf5: cell masks expanded by certain pixel
   - mask_mapped.hdf5: cell masks mapped to genes calls
   - **mask_trimmed.hdf5**: cell masks trimmed through removing repeated cells within overlap
   - cell_deleted.txt: repeated cells within overlap 
