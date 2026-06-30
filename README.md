# **BARseq3: a modular system for integrating spatial multi-omics and cellular barcoding in single cells**
Huihui Qi*, Manjari M-G Anant*, Dylan Z. Faltine-Gonzalez*, Ruitao Hu, Lai Wei, Christopher D. Workman, Caleb Shi, Ishbel Del Rosario, Justus M. Kebschull

## Folders:

### BARseq3_pipeline:
- gene_barcode_raw_image_pipeline: Pipeline to go from raw nd2 images to AnnData object & gene x cell x spatial matrices for transcriptomics + barcode experiments (Figure 2).
- gene_only_raw_image_pipeline: Pipeline to go from raw nd2 images to AnnData object & gene x cell x spatial matrices for transcriptomics only experiments (Figure 3).

### Figure2:
- BARseq2 and BARseq3 Amplicons/Cell analysis in Barcoded cells
- Specificity experiments: Pcp2 specificity experiment analysis
  
### Figure3:
- processed_data_figures.ipynb: Generating cell type analyses of AnnData object for Figure 3 and Supplementary Figure 4.
- raw_data_figures.ipynb: Generating analyses of raw data for Figure 3.

### Supplemental Figures:
- SupplFig1: BARseq2 and BARseq3 Amplicons/Cell analysis in non-barcoded cells and Vglut1 and Gad1 amplicons distribution analysis
- SupplFig3: Malat1 specificity experiment analysis


