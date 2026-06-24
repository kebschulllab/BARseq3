import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["JAVA_HOME"] ="/home/manjari/Downloads/Fiji.app/java/linux-amd64/zulu8.60.0.21-ca-fx-jdk8.0.322-linux_x64/jre/lib/amd64/server/"
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import napari
# import IPython.display
import bardensr
import bardensr.plotting
# import imagej
import copy
from datetime import datetime
import shutil
import imagej
from cellpose import plot
from cellpose import utils, io
import tifffile
import argparse
import warnings
warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import io as io
from . import preprocessing as preproc
from . import gene_calling as genecall
from . import cell_segmentation as cellseg
from .config import Config


import IPython.display
import PIL
from PIL import Image, ImageDraw

def plot_dots_brightness(config, col=5):
    image_registered = io.open_hdf5_NxRxC(config.filepath_output+'image_registered.hdf5', config.fov_num, config.round_num)
    gene_called = pd.read_csv(config.filepath_output+'gene_called.csv')
    # obtain intensity
    intensity_avg = []
    for FOV in range(config.fov_num):
        image_fov = np.max(image_registered[FOV], axis=1)
        gene_fov = gene_called[gene_called['FOV']==FOV]
        intensity = image_fov[:, gene_fov['m1'], gene_fov['m2']]
        intensity_avg.append(np.average(intensity, axis=1))
    intensity_avg = np.array(intensity_avg)

    # plot
    row = math.ceil(config.fov_num / col)
    plt.figure(figsize=(5*col, 5*row))
    for FOV in range(config.fov_num):
        plt.subplot(row, col, FOV+1)
        plt.plot(config.round_index, intensity_avg[FOV])
        plt.title('FOV %s' % str(FOV))
        plt.ylim(np.min(intensity_avg)-0.01, np.max(intensity_avg)+0.01)
        plt.xlabel('round')
        plt.ylabel('pixel value')
    plt.savefig(config.filepath_val+'dots_brightness_across_rounds.png')

def qc_genecalling(gene_result, genenames):
    nogene_names = [s for s in genenames if "nogene" in s]
    trgene_names = [s for s in genenames if "nogene" not in s]
    nogene_result = gene_result.iloc[np.where(np.isin(gene_result['Names'],nogene_names))]
    trgene_result = gene_result.iloc[np.where(np.isin(gene_result['Names'],trgene_names))]
    fdr_overall = (len(nogene_result)+0.01)/(len(np.unique(nogene_result['Names']))+0.01)*(len(np.unique(trgene_result['Names']))+0.01)/(len(trgene_result)+0.01)
    print(len(nogene_result), len(trgene_result))
    fdr_list, fov_list = [], []
    for i in range(int(np.unique(gene_result['FOV'])[-1])+1):
        nogene_result_fov = nogene_result[nogene_result['FOV']==i]
        trgene_result_fov = trgene_result[trgene_result['FOV']==i]
        fdr = (len(nogene_result_fov)+0.01)/(len(np.unique(nogene_result_fov['Names']))+0.01)*(len(np.unique(trgene_result_fov['Names']))+0.01)/(len(trgene_result_fov)+0.01)
        # if fdr < 0.8:
        fdr_list.append(fdr)
        fov_list.append(i)
    print(fdr_list)
    print(fov_list)
    # fdr_mean = np.mean(fdr_list)
    print(fdr_overall)
    return fov_list, fdr_list, fdr_overall

def plot_incorrect_genes(config, col=3):
    gene_called = pd.read_csv(config.filepath_output+'gene_called.csv')
    codebook, genenames, codeflat = genecall.create_codebook(config.filepath_codebook, config.round_index0, config.base_code)

    nogene_list = [s for s in genenames if "nogene" in s]
    percentage = []
    for NOGENE in nogene_list:
        percentage_nogene = []
        for FOV in range(config.fov_num):
            gene_fov = gene_called[gene_called['FOV']==FOV]
            gene_nogene = gene_fov[gene_fov['Names']==NOGENE]
            percentage_nogene.append((len(gene_nogene))/(len(gene_fov)+0.01)*100)
        percentage.append(percentage_nogene)

    percentage.append(np.sum(percentage, axis=0))
    nogene_list.append('all nogenes')

    row = math.ceil(config.fov_num / col)
    plt.figure(figsize=(col*10, row*5))
    for i in range(len(nogene_list)):
        plt.subplot(row, col, i+1)
        plt.bar(np.arange(config.fov_num), percentage[i])
        plt.title(nogene_list[i])
        plt.xlabel('fov')
        plt.ylabel('percentage of no (%)')
    plt.savefig(config.filepath_val+'incorrect_genes_percentage.png')

def plot_fdr(config):
    gene_called = pd.read_csv(config.filepath_output+'gene_called.csv')
    codebook, genenames, codeflat = genecall.create_codebook(config.filepath_codebook, config.round_index0, config.base_code)
    fov_list, fdr_list, fdr_overall = qc_genecalling(gene_called, genenames)
    plt.figure()
    plt.scatter(fov_list, fdr_list, color='lightblue')
    plt.ylim(0, np.max(fdr_list)+0.1)
    plt.axhline(fdr_overall, color='lightcoral')
    plt.title("FDR Across FOV")
    plt.xlabel('fov')
    plt.ylabel('fdr')
    plt.savefig(config.filepath_val+'fdr_across_fov.png')

def plot_gene_per_cell(config):
    gene_mapped = pd.read_csv(config.filepath_output+'gene_mapped.csv')
    
    gene2cell_fov, unigene2cell_fov = [], []
    for FOV in range(config.fov_num):
        gene_fov = gene_mapped[gene_mapped['FOV']==FOV]
        gene_cell, unigene_cell = [], []
        for CELL in np.unique(gene_fov['cell_number']):
            temp = gene_fov[gene_fov['cell_number']==CELL]
            gene_cell.append(len(temp))
            unigene_cell.append(len(np.unique(temp['Names'])))
        gene2cell_fov.append(np.average(gene_cell))
        unigene2cell_fov.append(np.average(unigene_cell))

    plt.figure()
    if int(config.fov_num*0.2) >5: plt.figure(figsize=(int(config.fov_num*0.2)*2, int(config.fov_num*0.2)))
    bar1 = plt.bar(x=np.arange(config.fov_num), height=gene2cell_fov, width=0.4, label='genes per cell', color='lightblue', tick_label=range(config.fov_num))
    bar2 = plt.bar(x=np.arange(config.fov_num)+0.4, height=unigene2cell_fov, width=0.4, label='unique gene per cell', color='lightcoral')
    # plt.bar_label(bar1)
    # plt.bar_label(bar2)
    plt.title('genes per cell & unique gene per cell')
    plt.xlabel('fov')
    plt.ylabel('num')
    plt.xticks(np.arange(config.fov_num)+0.2, range(config.fov_num))
    plt.legend()
    plt.savefig(config.filepath_val+'gene_per_cell_and_unique_gene_per_cell.png')