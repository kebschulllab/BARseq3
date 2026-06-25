# Get Started

## First, clone this repository
If logging into rockfish then 
- <code>ssh -Y login.rockfish.jhu.edu -l userid</code>
- then activate anaconda
- <code>module load anaconda</code>

Navigate to the directory you want to clone the STARmap repository into then clone the code under this branch:
- <code>git clone --single-branch --branch snakemake https://github.com/mmganant/STARmap_analysis.git</code>

If this method does not work then use an ssh to clone the repository:
- generate a rockfish key first using the methods described here: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=Linux
- Add the key to your github account using these instructions: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account.
- to copy the key open using <code>vim ~/.ssh/id_ed25519.pub</code> then copy it
- then clone the repository using <code>git clone --single-branch --branch snakemake git@github.com:mmganant/STARmap_analysis.git</code>

## Next, activate the Conda environment
### Create environment, if not already created
- use environment *starmap*: <code>conda create -n starmap python==3.9.2</code>
- activate the environment: <code>conda activate starmap</code>
- make sure CUDA, CUDNN, and NVIDIA/GPU Drivers are properly installed
- install tensorflow based on the system: https://www.tensorflow.org/install/pip (you may need to downgrade Numpy version)
   - this should be tensorflow version 2.18.0 on Rockfish!!
   - <code>python3 -m pip install tensorflow==2.18</code>
- install bardensr dependency:
   -<code>pip install --upgrade git+https://github.com/jacksonloper/bardensr.git</code>
   -**note: you must have CUDA properly downloaded and enabled for fast GPU processing
- install additional dependencies: <code>pip install n2v scanpy anndata cellpose pims-nd2 ipython imagej napari SimpleITK</code>
- pip install snakemake
- pip install pulp==2.5.1

## Running the the starmap analysis pipeline

### Each time you login into Rockfish, you need to activate the following modules and activate the environment in this order:
1. <code>module load anaconda</code>
2. <code>conda activate starmap</code>
3. <code>module load cuda/12.5.0 cudnn/9.2.0</code>

### If you want to use the config.json method of running the pipeline then modify the configurations for your dataset/filepaths/etc
- navigate to the folder containing the "config.json" document
- type <code>vi config.json</code> into the terminal to activate the text editor
- press <code>i</code> to insert text and change the configurations at the top of the file
- press <code>Esc</code>, then <code>:wq!</code> to write the changes to the document
- if you do not want to save your edits then just hit <code>:q!</code>
- ensure that the config.yaml file in profile folder found in the STARmap_analysis folder has the correct account name for the rockfish account that you are using.
- If you want to submit multiple sectioned jobs using run_all_configs_snakemake_nohup then ensure that the filepaths are correctly leading to your config directory.    

#### Once ready to run
- run <code>./run_starmap_snakemake_nohup.sh</code>
- this will write the output log to snakemake_output.log ... you can look at this log by typing <code>cat snakemake_output.log</code>

### You can also directly edit the Snakemake_nopatch file by commenting out the config paths and adding paths directly to your files.
- If you use this method then edit the Snakemake_nopatch file using the above vi method. If you use this method the config files and method will not work unless you edit back in the config paths.

#### If you use this method, then save the STARmap analysis directory within the experimental directory so that you can run mulitple sessions simultaneously, ex place copies of the STARmap analysis directory in the Section 1 directory and Section 2 directory. This will allow for one SLURM submission for Section 1 and another for Section 2

#### Once ready to run
- <code>nohup snakemake --slurm --jobs   "$(python -c 'import h5py; f=h5py.File("image_origin.hdf5","r"); print(int(len(f["list"])))')"   -s Snakefile_nopatch   --profile profile   --latency-wait 300   --rerun-incomplete   > snakemake_output.log 2>&1 &</code>
- change the "image_origin.hdf5" so that it is the path for your own "image_origin.hdf5" file. If for some reason that does not work then just type in the number of FOVs you are running:
   - <code>nohup snakemake --slurm --jobs 34 -s Snakefile_nopatch   --profile profile   --latency-wait 300   --rerun-incomplete   > snakemake_output.log 2>&1 &</code>

## Check code progress
the code should run fine. If you want to check the progress of it then type in <code>cat snakemake_output.log</code>

# File Hierarchies

## Run the script
necessary files for running the pipeline:  

└── ~~ data path (entered by user during configuration) ~~  
&emsp;&emsp;├── image_origin.hdf5           

&emsp;&emsp;├── codebook.csv  * can be named something else - must specify during config

&emsp;&emsp;├── dapi.npy 

&emsp;&emsp;├── positions_org.csv     

&emsp;&emsp;├── output/       * output created in the same folder

&emsp;&emsp;&emsp;&emsp;├── adata.h5ad


## Notes for config
|variables           |notes                                          |
| ------------------ | --------------------------------------------- |
|**path**|path to where output file is created|
|**gtca_path**|path to where image_origin.hdf5 is stored|
|**dapi_path**|path to where dapi_all_z is stored|
|**positions_path**|path to where positions are stored (*.csv)|
|**codebook_path**|path to where codebook is stored (*.csv)|
|**FOVs**|range of field of views|
|**FOV_minmax**|field of views used for computing min/max and find threshold|
|**filepath_dapi**|path to dapi img|
|**radius**|radius of background subtraction|
|**max_wiggle**|maximum displacement with bardensr registration|
|**niter**|number of iterations bardensr registration|
|**round_num**|total rounds [1 ,2 ,3, 4, 5, 6] start from 1 :exclamation: should include more than one round|
|**fov_align**|field of view used to perform channel alignment correction - change to field of view with as many dots as possible|
|cc_coeff|[zero_one, one_zero, two_three, three_two]|
|**fov_minmax**|specify the fovs for calculating the minimum and maximum values|
|radius|set the radius of background subtraction (larger -> less background subtracted)|
|base_code|the list of base in codebook|
|**thresh_refined**|threshold used for calling genes if find_threshold = False|
|**find_threshold**|find threshold (True) or use thresh_refined (False)|
|**precision**|precision of threshold if find_threshold = True|
|**output_cellboundaries**|outputs the segmentation boundaries in Vizgen format = True|
|**use_dapi_nissl_max**|use max projected DAPI and/or NISSL for cell segmentation = True|
|**expand_dist**|the number of pixels to expand the cell segmentation masks = 15|
|**use_cv2_blob_detector**|method to use for alignment, default is scimmage = False|
|**nogene_keyword**|keyword for your nogene used in FDR calculation = "nogene"|
|**fdr_FOVs_to_exclude**|FOVs to exclude from the average threshold calculation = []|


example:
path = "ma31_e13_b5/" #(Snakefile path)
gtca_path = path + "" #image_origin.hdf5
dapi_path = path + "" # dapi_all_z.npy
codebook_path = path + "CAMs_TFs_barcodes_final_8nogene_spikein.csv"
positions_path = path + "position_org.csv"
positions_glob = positions_path


FOVs = range(35)
FOV_minmax = np.array([10,20,5,7,17])
fov_align = 7
radius = 10
max_wiggle = 15
niter = 80
round_num =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
noisefloor = 0.05,
len_wid = 1000,
base_code = ["G", "T", "C", "A"]
fdr_thresh = 0.2
nogene_keyword = "no_gene"
thresh_refined = 0.75,
find_threshold = False
precision = 0.005
