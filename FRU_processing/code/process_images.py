#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 13:02:59 2018

@author: Estibaliz Gomez de Mariscal
Contact esgomezm@pa.uc3m.es for support.
"""

## Import the required packages.
#--------------------------------------------------------------------------------------------------
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
#import sys
#sys.path.insert(0,'/home/esgomezm/anaconda3/envs/3dprotucel/lib/python3.6/site-packages/SimpleITK/')
import SimpleITK as sitk
import skimage
import glob
import pandas
import time


from keras.models import load_model
from functions import unet_segmentation



## Set the required parameters.
#--------------------------------------------------------------------------------------------------
final_resolution = 1.566
dataset_num = 4
patch_size = 400
overlap = 125 
min_size = np.round(np.pi*((30/2)**2)/(final_resolution**2))
split_clusters = 1
min_hole_size = 25
top_hat_size = 10
sigma = 3
smooth_t = 0.4


min_size = np.round(np.pi*((30/2)**2)/(final_resolution**2))


## NOTE: update the paths to your machine
#--------------------------------------------------------------------------------------------------
# data_folder = 'data/dataset_01'
# data_folder = 'data/dataset_02'
data_folder = 'data/dataset_03'
output_path = 'fru_results/'

# Read the paths.
#--------------------------------------------------------------------------------------------------
resolution_file = '/resolution.xlsx'
path2im = glob.glob(data_folder + '/*.tif')
path2im.sort()
resolution = pandas.read_excel(data_folder + resolution_file)
resolution = np.array(resolution)

# Choose the model
#--------------------------------------------------------------------------------------------------

# Load FRU1.
# model = load_model('fully_residual_dropout_segmentation_03_04.h5')

# Load FRU2.
# model = load_model('fully_residual_dropout_segmentation_01_04.h5')

# Load FRU3.
model = load_model('fully_residual_dropout_segmentation.h5')

# Load model U1.
# model = load_model('unet_segmentation_3_4_reduce_learning_rate_400.h5')

# Load model U2.
# model = load_model('unet_segmentation_1_4_reduce_learning_rate_400.h5')

# Load model U3.
# model = load_model('unet_dropout_segmentation.h5')

# Load FRU trained only with Dataset 2.
# model = load_model('fully_residual_dropout_segmentation_03_nonoverfit.h5')


# Set tau threshold for probability maps
#--------------------------------------------------------------------------------------------------
threshold = 0.5


# Start image processing. 
#--------------------------------------------------------------------------------------------------
for f in range(len(path2im)):   
    start = time.time()
    bin_image, markers, im = unet_segmentation(path2im[f], dataset_num, resolution[f,1],
                            final_resolution, patch_size, threshold, overlap, min_size, model,
                            split_clusters, min_hole_size, top_hat_size, sigma, smooth_t)
    end = time.time()
    
    
    print("")
    print(path2im[f])
    print('Segmentation finished in ', (end-start), ' secs')
    with open("processing_time_FRU.txt", "a") as text_file:
        text_file.write("Frame %s: segmentation finished in %f" % (f, (end-start)))
       
    # Save the results as .tif files. 
    #-----------------------------------------------------------------------------------------------
    sitk.WriteImage(sitk.GetImageFromArray(bin_image.astype(np.int8)),output_path + path2im[f][-6:-4] +'_binary.tif')
    sitk.WriteImage(sitk.GetImageFromArray(markers.astype(np.int16)),output_path + path2im[f][-6:-4] +'_labels.tif')
    sitk.WriteImage(sitk.GetImageFromArray(im),output_path + path2im[f][-6:-4] +'.tif')
