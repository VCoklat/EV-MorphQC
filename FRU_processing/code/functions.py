#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:15:44 2018

@author: esgomezm
"""

# Load libraries and packages
import numpy as np
import SimpleITK as sitk
#from os import listdir
#import matplotlib.pyplot as plt
import scipy
#import scipy.misc
import skimage
#from skimage import filters
#from skimage import morphology
#from skimage.morphology import watershed, white_tophat, black_tophat
# from skimage.feature import peak_local_max
from skimage.morphology import square, disk
import PIL
from PIL import Image, ImageOps
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def images_resize(path, dataset, initial_resolution, final_resolution):
#    final_resolution = 1.566 

    aux = sitk.ReadImage(path, sitk.sitkUInt16)
    aux = sitk.GetArrayFromImage(aux)
    if dataset == 1:
        aux = aux[48:,48:]

    img = Image.fromarray(aux)

    imagesize_h = img.size[0]
    imagesize_w = img.size[1]
    hsize = int(np.round(float(imagesize_h) * float(initial_resolution)/final_resolution))
    wsize = int(np.round(float(imagesize_w) * float(initial_resolution)/final_resolution))

    img = img.resize((hsize, wsize))

    img = np.array(img)                 
    return img


def centroids(mask, radious):
    mask = mask.astype(np.int8)
    box = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2],1))
    for i in range(len(mask)):
        mask_itk = sitk.GetImageFromArray(mask[i,:,:,0])
        mask_connected_comp = sitk.ConnectedComponent(mask_itk)
        fil_label = sitk.LabelShapeStatisticsImageFilter()
        fil_label.Execute(mask_connected_comp)
        for l in fil_label.GetLabels():
            centroid = fil_label.GetCentroid(l)
    #        print i, l, centroid
            box[i,np.int(centroid[1]),np.int(centroid[0]), 0] = 1
            for x in range(fil_label.GetBoundingBox(l)[0],fil_label.GetBoundingBox(l)[0]+fil_label.GetBoundingBox(l)[2]):
                for y in range(fil_label.GetBoundingBox(l)[1],fil_label.GetBoundingBox(l)[1]+fil_label.GetBoundingBox(l)[3]):
                    if ((x-np.int(centroid[0]))**2 + (y-np.int(centroid[1]))**2) <= radious**2:
#                         box[i, y,x,0] = 0
#                     else:
                        box[i, y,x,0] = 1;       
    return box


def mirror_border(image, size): 
        
    im_size_h = image.shape[0] # height 
    im_size_w = image.shape[1] #width
    new_image = np.zeros((3*im_size_h,3*im_size_w))
    
    img = Image.fromarray(image.astype(np.float))
    mirror_img = PIL.ImageOps.mirror(img)
    mirror_img = np.array(mirror_img)
    
    new_image[im_size_h:2*im_size_h,:im_size_w] = mirror_img 
    new_image[im_size_h:2*im_size_h,im_size_w:2*im_size_w] = np.array(img)
    new_image[im_size_h:2*im_size_h,2*im_size_w:] = mirror_img 
    
    
    img = img.rotate(180)
    new_image[:im_size_h:,:im_size_w] = np.array(img)
    new_image[:im_size_h,2*im_size_w:] = np.array(img) 
    new_image[2*im_size_h:,:im_size_w] = np.array(img) 
    new_image[2*im_size_h:,2*im_size_w:] = np.array(img)
    
    mirror_img = PIL.ImageOps.mirror(img)
    mirror_img = np.array(mirror_img)
    new_image[:im_size_h,im_size_w:2*im_size_w] =mirror_img 
    new_image[2*im_size_h:,im_size_w:2*im_size_w] =mirror_img 
    
    index_h = int(np.round((new_image.shape[0]-size)/2))
    index_w = int(np.round((new_image.shape[1]-size)/2))
    new_image = new_image[index_h:index_h+size,index_w:index_w+size]
    
    return new_image

def linear_stretch(im, inf_lim, sup_lim):
    # perform a linerar stretching of the image intensity values. The function returns an image of type float. 
    new_im = np.copy(im)
    new_im = new_im.astype(np.float32)
    inf_im = np.min(new_im)
    sup_im = np.max(new_im)
    
    coef = (sup_lim - inf_lim)/(sup_im - inf_im)
    new_im = coef*(new_im-inf_im) + inf_lim
    
    return new_im
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# FINAL SEGMENTATIONS 
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def remove_obj_boundary(img):
    img = img.astype(np.int8)
    orImage = sitk.GetImageFromArray(img)
    image = np.copy(img)
    image[1:-1,1:-1] = 0
    sitkImage = sitk.GetImageFromArray(image)
    recons_filter = sitk.BinaryReconstructionByDilationImageFilter()
    sitkImage = recons_filter.Execute(sitkImage,orImage)
    image = sitk.GetArrayFromImage(sitkImage)
    final = img - image
    return final
def correct_cluster_separation(im, CI, min_size):
#     After routines for cluster separation it's common to obtain over segmentation of elements. 
#     This function analyses the resulting cluster-split of an object CI. 
#     Min_size is the minimum size required for the experiment: exosomes diameter is in a range (30,200)nanometers. 
    
    # Label all connected components.
    conncomp = sitk.GetImageFromArray(im)
    conncomp = sitk.ConnectedComponent(conncomp)
    fil_label = sitk.LabelShapeStatisticsImageFilter()
    fil_label.Execute(conncomp)
    conncomp = sitk.GetArrayFromImage(conncomp)
    labels = np.max(conncomp)
    l = 1
    while (l <= labels+1):
        aux = conncomp == l
        aux = aux.astype(np.int8)
        aux = sitk.GetImageFromArray(aux)
        aux = sitk.BinaryFillhole(aux) # After radon cluster split, holes may appear in the image. 
        aux = sitk.GetArrayFromImage(aux) 
        # Correct only too small objects
        if np.sum(aux) < min_size:
            # Obtain the labels nearby the object
            big_aux = skimage.morphology.binary_dilation(aux, square(5))
            big_aux = big_aux-aux
            side_labels = np.multiply(big_aux,conncomp)
            side_labels = np.unique(side_labels[side_labels>0],return_counts=False)
            if len(side_labels) == 1:
                final_l = side_labels
                aux_1 = conncomp == final_l
                aux = aux + aux_1
            else:
                k = 0
                final_l = labels+1
                for sl in range(len(side_labels)):
                    aux_1 = conncomp == side_labels[sl]
                    if np.sum(aux_1)>k:
                        k = np.sum(aux_1)
                        final_l = side_labels[sl]
                aux_1 = conncomp == final_l
                aux = aux + aux_1
            aux = skimage.morphology.binary_dilation(aux,disk(1))
            aux = np.multiply(aux,CI)
            aux = aux.astype(np.int8)
            aux = sitk.GetImageFromArray(aux)
            aux = sitk.BinaryFillhole(aux)
            aux = sitk.GetArrayFromImage(aux)
            # Substitute the new element.
            conncomp[aux == 1] = final_l
            
            # Update labels in connected components.
            for j in range(l+1,labels+1):
                conncomp[conncomp==j]=j-1
            labels = labels-1  
        else:
            conncomp[aux==1] = l
            l = l+1
    return conncomp

def radon_cluster_separation(seg, min_hole_size,top_hat_size,min_size):
#     This funtion analyses the sinogram of each connected component (object) in the image seg using the Radon transform. 
#     It calculates the local minima of the sinogram and reconstruct the corresponding segment of that local minima. 
#     All segments are the candidates to split a cluster. Then the cluster-split is post-processed with the function
#     correct_cluster_separation to discard oversegmentations. 

#     Parameters:
#     min_hole: restriction to accept the holes detected in the sinogram. Recommended 25 pixels
#     top_hat_size: size of the kernel to enhance the contrast in the sinogram. Recommended 10
#     min_size: minimum size (area in pixels) required for the experiment: exosomes diameter is in a range (30,200)nanometers.

    final_im = np.zeros(seg.shape)
    final_im = final_im.astype(np.int8)
#    final_im_os = np.copy(final_im)
    counter = 1
    for j in range(1 ,np.max(seg)+1):
        seg_bin = seg == j
        seg_bin = seg_bin.astype(np.uint8)
        
        if np.sum(seg_bin)>= 2*min_size:            
            image = sitk.GetImageFromArray(seg_bin)
            image = sitk.ConnectedComponent(image)
            fil_label = sitk.LabelShapeStatisticsImageFilter()
            fil_label.Execute(image)
            for l in fil_label.GetLabels():
                BB = fil_label.GetBoundingBox(l)
            CI = seg_bin[BB[1]-1:BB[1]+BB[3]+1, BB[0]-1:BB[0]+BB[2]+1]
            CI_aux = skimage.morphology.erosion(CI,square(4)) 
            theta = np.linspace(0., 180., max(CI.shape), endpoint=False)
            sinogram = skimage.transform.radon(CI_aux, theta=theta, circle = False)
            # Enhance sinogram's edges and intensity differences. 
            new_sinogram = sinogram + skimage.morphology.white_tophat(sinogram,disk(top_hat_size)) - skimage.morphology.black_tophat(sinogram,disk(top_hat_size))
            val = skimage.filters.threshold_otsu(new_sinogram)
            sinogram_image = new_sinogram > val
            sinogram_image = sinogram_image.astype(np.int)
            
            # It may happen that inner holes touch image edges in the sinogram. 
            # Therefore we need to ensure that those wholes won't be removed.
            
            mask = 1-sinogram_image
            mask[:,0] = 0
            mask[:,-1] = 0            
            sinogram_hole = remove_obj_boundary((mask))
            
            # Obtain the local minima of each hole. 
            aux_im = sitk.GetImageFromArray(sinogram_hole)
            aux_im = sitk.ConnectedComponent(aux_im)
            fil_label = sitk.LabelShapeStatisticsImageFilter()
            fil_label.Execute(aux_im)
            aux_im = sitk.GetArrayFromImage(aux_im)
            group_sep_line = np.zeros(CI.shape)
            for l in fil_label.GetLabels():                   
                if np.sum(aux_im == l) >= min_hole_size:
                    aux = np.min(sinogram[aux_im == l])
                    sinogram_hole = np.multiply(sinogram, aux_im == l)
                    # We want a line for the reconstruction. 
                    
                        # The point in the sinogram represents the projection of the pixels that lie in a line: 
                        # the sum of their values. 
                        # Therefore the projection of a complete segment of value 1 in all its pixels will result 
                        # in the length of the diagonal of the image == rows in the sinogram: sinogram.shape[0].
                    
                    reconstruction_fbp = skimage.transform.iradon(sinogram.shape[0]*(sinogram_hole == aux), theta=theta, circle=False)
                    
                    # Adapt the reconstruction to the Region of Interest (ROI) CI.
                    row_l = np.int((reconstruction_fbp.shape[0]-CI.shape[0])/2)
                    col_l = np.int((reconstruction_fbp.shape[1]-CI.shape[1])/2)
                    im = np.zeros(reconstruction_fbp.shape)
                    if row_l == 0 and col_l == 0:
                        im[-CI.shape[0]:, -CI.shape[1]:] = CI
                    elif row_l == 0:
                        im[-CI.shape[0]:, -col_l-CI.shape[1]: -col_l] = CI
                    elif col_l == 0:
                        im[-row_l-CI.shape[0]:-row_l, -CI.shape[1]:] = CI
                    reconstruction_fbp = np.absolute(reconstruction_fbp)                    
                    reconstruction_fbp = np.abs(reconstruction_fbp) > 0.3
                    reconstruction_fbp = skimage.morphology.binary_dilation(reconstruction_fbp, disk(5))
                    reconstruction_fbp = skimage.morphology.thin(reconstruction_fbp)
                    im = np.multiply(im,reconstruction_fbp)
                    
                    if np.sum(np.absolute(im)) > 0:
                        sep_line = im > 0
                    else:
                        sep_line = im
                        
                    if row_l == 0 and col_l == 0:
                        sep_line = sep_line[-CI.shape[0]:, -CI.shape[1]:]
                    elif row_l == 0:
                        sep_line = sep_line[-CI.shape[0]:, -col_l-CI.shape[1]: -col_l]
                    elif col_l == 0:
                        sep_line = sep_line[-row_l-CI.shape[0]:-row_l,  -CI.shape[1]:]    
                    group_sep_line = group_sep_line + sep_line
                    
            # Separate the cluster and correct oversegmentation        
            sep_line = group_sep_line > 0    
            im = np.multiply(CI, 1-sep_line.astype(np.int))            
            im = np.multiply(CI, 1-sep_line.astype(np.int))
            im = correct_cluster_separation(im,CI, min_size)
            aux = np.zeros(im.shape)
            for sub_labels in range(1,np.max(im)+1):
                aux[im == sub_labels] = counter
                counter = counter + 1
                
            final_im[BB[1]-1:BB[1]+BB[3]+1, BB[0]-1:BB[0]+BB[2]+1] = final_im[BB[1]-1:BB[1]+BB[3]+1, BB[0]-1:BB[0]+BB[2]+1] + aux
        else:
            seg_bin[seg_bin == 1] = counter
            counter = counter + 1
            final_im = final_im + seg_bin
        # Return an image with labels. 
        final_im = final_im.astype(np.int8)
        
    return final_im

def GetMaskFromProbMap(prob_map, threshold, min_size, split_clusters, min_hole_size, top_hat_size):
    
    # Write the image in the sitk format to obtain the first segmentation
    mask_itk = sitk.GetImageFromArray(prob_map)
    
    # Threshold the probability map and erode it in orther to eliminate possible small points
    filter_itk = sitk.BinaryThresholdImageFilter()
    filter_itk.SetLowerThreshold(threshold) # threshold = 0.15
    mask_itk = filter_itk.Execute(mask_itk)
    mask_itk = sitk.BinaryErode(mask_itk,2) # 2
    mask_itk = sitk.BinaryDilate(mask_itk,2) # 2
    mask_itk = sitk.BinaryFillhole(mask_itk)
    image = sitk.GetArrayFromImage(mask_itk)
    
    # Remove cell bodies touching the edges. 
    image = remove_obj_boundary(image)
    mask_itk = sitk.GetImageFromArray(image)
#     # Calculate the distance transform for watershed implementation
#     distance = ndi.distance_transform_edt(image)
    
#     # Smooth the distance transform to avoid small blobs
#     sitkDistance = sitk.GetImageFromArray(distance)
#     filter_Gauss = sitk.DiscreteGaussianImageFilter()
#     filter_Gauss.SetVariance(15)
#     filter_Gauss.SetMaximumKernelWidth(1)
#     fil_label = filter_Gauss.Execute(sitkDistance)
#     distance = sitk.GetArrayFromImage(fil_label)
    
#     # get the markers for the Watershed segmentation
#     local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((2,1)), labels=image)
#     markers = ndi.label(local_maxi)[0]
    
    mask_itk =sitk.ConnectedComponent(mask_itk)
    label_id = sitk.LabelShapeStatisticsImageFilter()
    label_id.Execute(mask_itk)
    markers = sitk.GetArrayFromImage(mask_itk)
        
    # Ensure a minimum size
    k = np.max(markers)
    i = 1
    while (i <= k+1):
        s = markers[markers==i].shape
        # print i, s[0], np.max(markers)
        if s[0] < min_size:
            markers[markers==i] = 0
            for j in range(i+1,k+1):
                markers[markers==j]=j-1
            k = k-1
        else:
            i = i+1
        # print i, markers[markers==i].shape
    if split_clusters == 1:
        markers = radon_cluster_separation(markers, min_hole_size,top_hat_size,min_size)
    
    image = markers > 0
    image = image.astype(np.int8)
    
    return markers, image

def obtain_patches_newdata(im, patch_size, overlap):
#     Images have different resolutions and therefore, different sizes. 
#     For an efficient data augmentation, the network was fed with patches.
#     The patches should be big enough as to have at least one exosome inside
#     and its size has to coincide with convolutional network's restrictions 
#     for down- and up-samplings.
#     This function is called complete because it takes patches of all images 
#     regardless their size, so images are segmented completely
# radious = 8
# patch_size = 500
# overlap = 125
    patches_500 = np.zeros((1,patch_size,patch_size,1))
    imagesize_h = int(im.shape[0]) # height of the image
    imagesize_w = int(im.shape[1]) # width of the image
    
    if imagesize_h < patch_size and imagesize_w < patch_size:
        aux = mirror_border(im, patch_size)
#        aux = aux.reshape((1,aux.shape[0],aux.shape[1],1))
#        patches_500 = np.concatenate((patches_500,aux))
    elif imagesize_h < patch_size and imagesize_w >= patch_size:
        aux = mirror_border(im, imagesize_w)
        index_h = int(np.round((aux.shape[0]-patch_size)/2))
        aux = aux[index_h:index_h+patch_size]
    elif imagesize_h >= patch_size and imagesize_w < patch_size:
        aux = mirror_border(im, imagesize_h)
        index_w = int(np.round((aux.shape[1]-patch_size)/2))
        aux = aux[:imagesize_h,index_w:index_w+patch_size]
    else:
        aux = im
    imagesize_h = int(aux.shape[0]) # height of the image
    imagesize_w = int(aux.shape[1]) # width of the image    
    dist = int(patch_size - overlap)
    patch_n_h = int(np.floor((imagesize_h - patch_size)/dist))
    patch_n_w = int(np.floor((imagesize_w - patch_size)/dist))
    for nr in range((patch_n_h+1)):
        for nc in range((patch_n_w+1)):
            patch = aux[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size]
            patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
            patches_500 = np.concatenate((patches_500,patch))
    if (imagesize_w - patch_size) % dist != 0:
        for i in range((patch_n_h+1)):
            # Rows
            patch = aux[i*dist:i*dist + patch_size,-patch_size:]
            patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
            patches_500 = np.concatenate((patches_500,patch))
    if (imagesize_h - patch_size) % dist != 0:
        for i in range((patch_n_w+1)):
            # Column
            patch = aux[-patch_size:,i*dist:i*dist + patch_size]
            patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
            patches_500 = np.concatenate((patches_500,patch))
    if ((imagesize_h - patch_size) % dist != 0) and ((imagesize_w - patch_size) % dist != 0): 
        # Diagonal
        patch = aux[-patch_size:,-patch_size:]
        patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
        patches_500 = np.concatenate((patches_500,patch))                                
    patches_500 = patches_500[1:]
    patches_500 = patches_500.astype(np.float32)
        
    return patches_500

def reconstruct_frame_complete(patches_im, imagesize_h, imagesize_w, patch_size, overlap):
#     Images have different resolutions and therefore, different sizes. We want to analyze patches so the training data
#     can be augmented more efficiently. The patches should be big enough as to have at least one exosome inside and 
#     its size has to coincide with convolutional network's restrictions for down- and up-samplings.

#     For the training, when the images were smaller than 800x800 we only took one patch so as to regret the 
#     redundancy of the data. Now, this fact is not taken into account as all the information in the image
#     is required for practical uses. Therefore, we call reconstruct_frame_complete because the function will reconstruct the 
#     image from its patches completely, regardless its size. 
# patch_size = 400
# overlap = 125
    
    aux = None # Initialize the variable
    recons_frame = np.zeros((imagesize_h, imagesize_w))
    
    
    if imagesize_h < patch_size and imagesize_w < patch_size:
        index_h = int((patch_size-imagesize_h)/2)
        index_w = int((patch_size-imagesize_w)/2)
        recons_frame = patches_im[0,index_h:-index_h, index_w:-index_w,0]
        
    elif imagesize_h < patch_size and imagesize_w >= patch_size:
        index_h = int((patch_size-imagesize_h)/2)
        aux = patches_im[:,index_h:-index_h]
    elif imagesize_h >= patch_size and imagesize_w < patch_size:
        index_w = int((patch_size-imagesize_w)/2)
        aux = patches_im[:,:,index_w:-index_w]
    else:
        aux = patches_im
    
    if aux is not None:
        
        # distance between patches
        dist = int(patch_size - overlap)
        # count the number of opatches to do in each direction
        patch_n_h = int(np.floor((imagesize_h - patch_size)/dist))
        patch_n_w = int(np.floor((imagesize_w - patch_size)/dist))
        # The template will be used to weight the overlap 
        #(value 1 for non overlapping areas and value n where n areas overlap)
        template = np.zeros((imagesize_h, imagesize_w)) 
        p = 0;
        for nr in range((patch_n_h+1)):
            for nc in range((patch_n_w+1)):
                template[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size] = template[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size]+1
                recons_frame[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size] = recons_frame[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size] + aux[p,:,:,0]
                p = p+1

        if (imagesize_w - patch_size) % dist != 0:
            for i in range((patch_n_h+1)):
                # Rows
                template[i*dist:i*dist + patch_size,-patch_size:] = template[i*dist:i*dist + patch_size,-patch_size:] + 1
                recons_frame[i*dist:i*dist + patch_size,-patch_size:] = recons_frame[i*dist:i*dist + patch_size,-patch_size:] + aux[p,:,:,0]
                p = p+1
        if (imagesize_h - patch_size) % dist != 0:
            for i in range((patch_n_w+1)):
                # Column
                template[-patch_size:,i*dist:i*dist + patch_size] = template[-patch_size:,i*dist:i*dist + patch_size] + 1
                recons_frame[-patch_size:,i*dist:i*dist + patch_size] = recons_frame[-patch_size:,i*dist:i*dist + patch_size] + aux[p,:,:,0]
                p = p+1
        if ((imagesize_h - patch_size) % dist != 0) and ((imagesize_w - patch_size) % dist != 0):
            # Diagonal
            template[-patch_size:,-patch_size:] = template[-patch_size:,-patch_size:] + 1
            recons_frame[-patch_size:,-patch_size:] = recons_frame[-patch_size:,-patch_size:] + aux[p,:,:,0]
            p = p+1
        template = 1/template
        recons_frame = np.multiply(recons_frame,template)
    return recons_frame

def images_resize2original(path, im, dataset):
#    final_resolution = 1.566 

    originalI = sitk.ReadImage(path, sitk.sitkUInt16)
    aux = sitk.GetArrayFromImage(originalI)
    if dataset == 1:
        aux = aux[48:,48:]

    originalIsize_h = aux.shape[0]
    originalIsize_w = aux.shape[1]
    del aux
    
    im = Image.fromarray(im)
    im = im.resize((originalIsize_w, originalIsize_h))

    im = np.array(im)                 
    return im

def unet_segmentation(path2im, dataset_num, original_resolution, final_resolution, patch_size, threshold, overlap, min_size, model, split_clusters, min_hole_size, top_hat_size, sigma, smooth_t):
    im = images_resize(path2im, dataset_num, original_resolution, final_resolution)
    
    patches_im = obtain_patches_newdata(im, patch_size, overlap)
    for p in range(patches_im.shape[0]):
        patches_im[p] = linear_stretch(patches_im[p], 0, 1)
    patches_im = patches_im.astype(np.float32)
    for p in range(patches_im.shape[0]): 
        aux = patches_im[p]
        aux = aux.reshape((1, aux.shape[0],aux.shape[1],aux.shape[2]))
        aux = model.predict(aux)
        patches_im[p] = aux[0] 
        print(p)
    recons_frame = reconstruct_frame_complete(patches_im, im.shape[0],  im.shape[1], patch_size, overlap)
    del(patches_im)
    
    markers, bin_image = GetMaskFromProbMap(recons_frame, threshold, min_size,split_clusters, min_hole_size, top_hat_size)
    del(recons_frame)
    markers = images_resize2original(path2im, markers, dataset_num)
    
    markers_tps = np.zeros(markers.shape)
    for l in range(np.max(markers)):
        b_round = skimage.filters.gaussian(np.float16(markers==l+1), sigma=sigma, 
                                           mode = 'mirror', preserve_range=True)	
        b_round = np.int8(b_round>smooth_t)
        
        markers_tps[b_round == 1] = (l+1)
    bin_image = markers_tps>0
    markers_tps = np.int16(markers_tps)
    return bin_image, markers_tps, im

###############################################################################
################### Use this function if GPU available ########################
###############################################################################

#def unet_segmentation(path2im, dataset_num, original_resolution, final_resolution, patch_size, threshold, overlap, min_size, model, split_clusters, min_hole_size, top_hat_size):
##     data_folder = '/home/esgomezm/Projects/Exosomes_Martin/data/' + dataset
##     annotations_folder = '/home/esgomezm/Projects/Exosomes_Martin/annotations/' + dataset
##     resolution_file = '/resolution.xlsx'
##     path2im = glob.glob(data_folder + '/*.tif')
##     path2im.sort()
##     path2gt = glob.glob(annotations_folder + '/*.tif')
##     path2gt.sort()
##     resolution = pandas.read_excel(data_folder+ resolution_file)
##     resolution = np.array(resolution)
#    im = images_resize(path2im, dataset_num, original_resolution, final_resolution)
#    patches_im = obtain_patches_newdata(im, patch_size, overlap)
#    for p in range(patches_im.shape[0]):
#        patches_im[p] = linear_stretch(patches_im[p], 0, 1)
#    patches_im = patches_im.astype(np.float32)
#    patches_im = model.predict(patches_im)  
#    recons_frame = reconstruct_frame_complete(patches_im, im.shape[0], patch_size, overlap)
#    markers, bin_image = GetMaskFromProbMap(recons_frame, threshold, min_size,split_clusters, min_hole_size, top_hat_size)
#    
#    bin_image = images_resize2original(path2im, bin_image, dataset_num)
#    markers = images_resize2original(path2im, markers, dataset_num)
#    markers_tps = np.zeros(markers.shape)
#    for l in range(np.max(markers)):
#        b_round = skimage.filters.gaussian(np.float16(markers==l+1), sigma=sigma, 
#                                           mode = 'mirror', preserve_range=True)	
#        b_round = np.int8(b_round>smooth_t)
#        
#        markers_tps[b_round == 1] = (l+1)
#    bin_image = markers_tps>0
#    markers_tps = np.int16(markers_tps)
#    
#    return bin_image, markers_tps


###############################################################################
################## Watershed processing for cluster split #####################
###############################################################################

#def watershed_topbottom_markers(im, seg, medfilt_rad, transforms_rad, dil_rad, min_size):
#    # im              original image
#    # seg             image segmentation
#    # medfilt_rad     radii of the median filter. Recommended 10
#    # transforms_rad  radii for the disk of the transformation. Recommended 50
#    # dil_rad         radii of the dilatation after Otsu's threshold. Recommended 5
#    
#    im = linear_stretch(im, 0, 1)
#    im = medfilt2d(im, kernel_size = medfilt_rad)
#
#    TH = white_tophat(im, disk(transforms_rad))
#    TH = linear_stretch(TH, 0,1)
#    BH = black_tophat(im, disk(transforms_rad))
#    BH = linear_stretch(BH, 0, 1)
#    contrastEnhanced = im + TH - BH
#    
#    val = skimage.filters.threshold_otsu(contrastEnhanced)
#    markers = contrastEnhanced >= val
#    markers = markers.astype(np.int8)
#    bin_seg = seg > 0
#    filtIm = np.multiply(markers, bin_seg.astype(np.int8))
#
#    filtIm = skimage.morphology.binary_dilation(filtIm, disk(dil_rad))
#    filtIm_sitk = sitk.GetImageFromArray(filtIm.astype(np.int8))
#    filtIm_sitk = sitk.BinaryFillhole(filtIm_sitk)
#    filtIm_sitk = sitk.ConnectedComponent(filtIm_sitk)
#    fil_label = sitk.LabelShapeStatisticsImageFilter()
#    fil_label.Execute(filtIm_sitk)
#    markers = sitk.GetArrayFromImage(filtIm_sitk)
#    # Ensure a minimum size
#    k = np.max(markers)
#    i = 1
#    while (i <= k+1):
#        s = markers[markers==i].shape
#        # print i, s[0], np.max(markers)
#        if s[0] < min_size:
#            markers[markers==i] = 0
#            for j in range(i+1,k+1):
#                markers[markers==j]=j-1
#            k = k-1
#        else:
#            i = i+1
#    
#    # # # process the watershed
#    labels = watershed(contrastEnhanced, markers, mask = bin_seg, watershed_line=True)
#    return labels, markers


