# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:14:39 2022

@author: peins
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
import pickle
import sys
from tqdm import tqdm

import cv2
from skimage import filters, measure, morphology


import feature_extraction_lib as fel
import get_dataset_info as gdi


plt_ind = 0

def get_num_from_folder_name(folder_name, spheroid_format, condition_str, exp_num = 0):
    temp1 = re.findall(spheroid_format, folder_name)
    if len(temp1) ==1:
        sp_num = int(temp1[-1][1:])
    else:
        sp_num = int(-1)
        
    temp2 = np.array([len(re.findall(a, folder_name)) for a in condition_str.keys()])
    
    if sum(temp2) ==1:
        ind = np.where(temp2 ==1)[0][0]
        key1 = list(condition_str.keys())[ind]
        temp3 = condition_str[key1]
        condition_num1 = temp3[0]
        condition_num2 = temp3[1]
    else:
        print(folder_name, 'err_to_get_rec_ind')
        condition_num1 = -1
        condition_num2 = -1
    
    rec_vec = np.array([exp_num, condition_num1, condition_num2, sp_num])
    # print(rec_vec)
    
    return rec_vec

def get_png_mask_match(path1, png_folder, mask_folder, png_format):
    
    png_list = glob.glob(os.path.join(path1, png_folder, png_format))
    png_list.sort()
    # mask_list = glob.glob(os.path.join(folder_name, mask_folder, png_format))
    # mask_list.sort()
    
    match_list = list()
    
    for png1 in png_list:
        file_name = os.path.basename(png1)
        temp1 = os.path.join(path1, mask_folder, file_name)
        if os.path.isfile(temp1):
            match_list.append([png1, temp1])
            
    # import pdb;pdb.set_trace()
    return match_list
            
    
def get_img_mask_rec_form_dataset_info(dataset_info, ex_day = 0, base_path = './'):
    
    target_path = dataset_info['target_path']
    condition_str = dataset_info['condition_str']
    png_folder = dataset_info['png_folder'] 
    png_format = dataset_info['png_format'] 
    mask_folder = dataset_info['mask_folder']     
    spheroid_format = dataset_info['spheroid_format']
    exception_file = dataset_info['exception_file']
    remove_object = dataset_info['remove_object']
    remove_hole = dataset_info['remove_hole']
    
    path_dict_ori = {ex_day:glob.glob(os.path.join(base_path, a)) for a in target_path}
    path_dict = dict()
    for ii, (ex_day, path_list1) in enumerate(path_dict_ori.items()):
        path_list1 = [a for a in path_list1 if os.path.isdir(os.path.join(a, png_folder)) and os.path.isdir(os.path.join(a, mask_folder)) and not os.path.isfile(os.path.join(a, exception_file))]
        path_list1.sort()
        path_dict[ex_day] = path_list1

    
    rec_ind = list()
    prop_arr = list()
    contour_arr = list()
    for ii, (ex_day, path_list1) in enumerate(path_dict.items()):
        for jj, path1 in tqdm(enumerate(path_list1), total = len(path_list1)):
            folder_name = os.path.basename(path1)
            rec_vec = get_num_from_folder_name(folder_name, spheroid_format, condition_str, exp_num = ex_day)
            match_list = get_png_mask_match(path1, png_folder, mask_folder, png_format)
            
            for png_path1, mask_path1 in match_list:
                file_name = os.path.basename(png_path1)
                temp1 = re.findall('\d+', file_name)
                fr_num = int(temp1[-1])
                # print(fr_num)
                
                png1 = cv2.imread(png_path1, cv2.IMREAD_GRAYSCALE)
                mask1 = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
                
                threshold = filters.threshold_otsu(mask1)
                mask2 = mask1 > threshold
                mask2 = morphology.remove_small_objects(mask2, remove_object)
                mask2 = morphology.remove_small_holes(mask2, remove_hole)
                labels = measure.label(mask2)
                uni_labels = np.unique(labels)
                if uni_labels.shape[0] >1:
                    sim_area = [np.sum(labels == uni_label1) for uni_label1 in uni_labels.tolist() ]
                    max_ind = np.argmax(sim_area[1:]) +1
                    mask3 = labels == max_ind
                    mask3 = morphology.remove_small_holes(mask3, remove_hole)
                    
                    prop_arr1 = fel.get_region_prop2(mask3, png1)                    
                    contour_arr1 = fel.binary_mask_to_reduced_contour(mask3, contour_dim = 256, major_rot = True)
                    
                    
                    rec_ind1 = [*rec_vec, fr_num]
                    rec_ind.append(rec_ind1)
                    prop_arr.append(prop_arr1)
                    contour_arr.append(contour_arr1)
                else:
                    rec_ind1 = [*rec_vec, fr_num]
                    print(rec_ind1, 'error')
                    import pdb;pdb.set_trace()
    

    rec_ind = np.stack(rec_ind, axis=0)
   
    prop_arr = np.stack(prop_arr, axis=0)
    contour_arr = np.stack(contour_arr, axis=0)
    
    return rec_ind, prop_arr, contour_arr
    


if __name__ == '__main__':
    
    base_path = gdi.base_path
    # dataset_info_all = {0: gdi.dataset0_info, 1: gdi.dataset1_info, 2: gdi.dataset2_info}    
    dataset_info_all = {0: gdi.dataset0_info}
    
    rec_ind, prop_arr, contour_arr = list(), list(), list()    
    for ex_day, dataset_info in dataset_info_all.items():
        rec_ind1, prop_arr1, contour_arr1 = get_img_mask_rec_form_dataset_info(dataset_info, ex_day = ex_day, base_path = base_path)
        rec_ind.append(rec_ind1)
        prop_arr.append(prop_arr1)
        contour_arr.append(contour_arr1)
        
    rec_ind = np.concatenate(rec_ind, axis=0)
    prop_arr = np.concatenate(prop_arr, axis=0)
    contour_arr = np.concatenate(contour_arr, axis=0)

    pca_arr, pca_reducer = fel.fourier_contour_feature_pca(contour_arr, dim = 200)
    power_spec_arr = fel.power_spectrum_feature(contour_arr)
    power_norm_arr = fel.spf.get_norm_spectral_power_from_contour_arr(contour_arr)
    
    # xf = np.fft.fft(contour_arr[:,:,1], axis = -1)
    # yf = np.fft.fft(contour_arr[:,:,0], axis = -1)
    # freq = np.fft.fftfreq(contour_arr.shape[1])


    data_save_path = './data_save' + str(list(dataset_info_all.keys()))
    os.makedirs(data_save_path, exist_ok= True)
    np.save(os.path.join(data_save_path, 'rec_ind.npy'), rec_ind)
    np.save(os.path.join(data_save_path, 'prop_arr.npy'), prop_arr)
    np.save(os.path.join(data_save_path, 'contour_arr.npy'), contour_arr)
    np.save(os.path.join(data_save_path, 'pca_arr.npy'), pca_arr)
    np.save(os.path.join(data_save_path, 'power_spec_arr.npy'), power_spec_arr)
    np.save(os.path.join(data_save_path, 'power_norm_arr.npy'), power_norm_arr)
    
    pca_reducer_path = os.path.join(data_save_path, 'pca_reducer.pickle')
    with open(pca_reducer_path, 'wb') as pickle_file:
        pickle.dump(pca_reducer, pickle_file)
    
    
    fel.sigma_plot(pca_arr, pca_reducer, pca_axis = np.arange(20), save_path = os.path.join(data_save_path, 'pca_feature_result'))
    fel.recon_test1(contour_arr, pca_arr, pca_reducer, dim_lim = 20, index = np.arange(0, 500, 50), save_path = os.path.join(data_save_path, 'pca_feature_result'))




    