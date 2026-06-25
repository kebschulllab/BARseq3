import os
import sys
import pandas as pd
from datetime import datetime

from starmap import io as io
from starmap import cell_segmentation as cellseg


if __name__ == '__main__':

    filepath_homedir = sys.argv[1]
    position_file = sys.argv[2]
    dim = int(sys.argv[3])
    filepath_loadimg = filepath_homedir+"STARmap_loadImg/"
    filepath_output = filepath_homedir+"STARmap_output/"

    position_reg = pd.read_csv(filepath_loadimg + position_file + '.csv')
    gene_mapped = pd.read_csv(filepath_output + 'gene_mapped.csv')
    mask_mapped = io.open_hdf5(filepath_output + 'mask_expanded.hdf5')

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "----remove repeated gene spots within overlap area")
    gene_trimmed, mask_trimmed, cell_deleted = cellseg.overlap_rmv_mp(gene_mapped, mask_mapped, position_reg, dim)
    gene_trimmed.to_csv(filepath_output+'gene_trimmed.csv', index=False)
    io.save_hdf5(mask_trimmed, filepath_output+'mask_trimmed.hdf5')
    cell_deleted = pd.DataFrame(cell_deleted)
    cell_deleted.to_csv(filepath_output+'cell_deleted.csv', index=False)
