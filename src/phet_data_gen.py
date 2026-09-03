# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:50:05 2024

@author: peins
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scipy
import os
import copy

import sys
sys.path.append('../lib')
import feature_extraction_lib as fel
import get_dataset_info as gdi




plt_ind = 100

title_list = ['filled_area', 'major_axis_len', 'minor_axis_len', 'perimeter', 'convex_area', 
              'eccentricity', 'extend', 'solidity', 'circularities', 'aspect_ratio', 'orientation',
              'inten_mean', 'inten_std']


def condition_sel(data_arr1, cond_list = [], sum_type = 'or'):
    cond_ind_list = [data_arr1 == a for a in cond_list]
    cond_arr = np.stack(cond_ind_list, axis=1)
    sum_cond = np.max(cond_arr, axis=1)
    
    if sum_type in 'or':
        result_ind = sum_cond >0
        
    elif sum_type in 'and':
        result_ind = sum_cond >= len(cond_list)
    
    return result_ind

def save_pd_csv(file_name, data_dict):
    temp_pd =  pd.DataFrame(data = data_dict)
    temp_pd.to_csv(file_name, index= False)
    
def get_value_from_manual_masking(rec_ind, manual_mask_pd, pd_rec_ind = 'rec_ind', pd_masking = 'is_invasive'):
    
    rec_ind_keys = [key for key in manual_mask_pd.keys() if pd_rec_ind in key]
    rec_ind_keys.sort()
    len_rec = len(rec_ind_keys)
    
    masking_keys = [key for key in manual_mask_pd.keys() if pd_masking in key]
    masking_keys.sort()
    len_mask = len(masking_keys)
    
    pd_keys = rec_ind_keys + masking_keys
    
    pd_data = manual_mask_pd[pd_keys].to_numpy()
    
    result = -np.ones([rec_ind.shape[0], len_mask])
    for pd_data1 in pd_data:
        rec_ind1 = pd_data1[:len_rec]
        ind = fel.rec_to_ind(rec_ind, rec_ind1)
        result[ind] = pd_data1[-len_mask:]
    
    assert (all(result > -1))
    
    return result
        
    
    
    
    


if __name__ == '__main__':

    att_num = 1    
    data_save_path_dict = { 1:['./data_save[1]'], }

    rec_ind_list = list()
    prop_arr_list = list()
    contour_arr_list = list()
    pca_arr_list = list()
    power_spec_arr_list = list()
    power_norm_arr_list = list()
    major_ind_list = list()

    for data_save_path in data_save_path_dict[att_num]:
        rec_ind1 = np.load(os.path.join(data_save_path, 'rec_ind.npy'))
        prop_arr1 = np.load(os.path.join(data_save_path, 'prop_arr.npy'))
        contour_arr1 = np.load(os.path.join(data_save_path, 'contour_arr.npy'))
        pca_arr1 = np.load(os.path.join(data_save_path, 'pca_arr.npy'))
        power_spec_arr1 = np.load(os.path.join(data_save_path, 'power_spec_arr.npy'))
        power_norm_arr1 = np.load(os.path.join(data_save_path, 'power_norm_arr.npy'))

        manual_mask_pd = pd.read_csv(os.path.join(data_save_path, 'manual_marking.csv'))
        major_ind1 = get_value_from_manual_masking(
            rec_ind1,
            manual_mask_pd,
            pd_rec_ind='rec_ind',
            pd_masking='is_invasive',
        )[:, 0]

        rec_ind_list.append(rec_ind1)
        prop_arr_list.append(prop_arr1)
        contour_arr_list.append(contour_arr1)
        pca_arr_list.append(pca_arr1)
        power_spec_arr_list.append(power_spec_arr1)
        power_norm_arr_list.append(power_norm_arr1)
        major_ind_list.append(major_ind1)

    rec_ind = np.concatenate(rec_ind_list, axis=0)
    prop_arr = np.concatenate(prop_arr_list, axis=0)
    contour_arr = np.concatenate(contour_arr_list, axis=0)
    pca_arr = np.concatenate(pca_arr_list, axis=0)
    power_spec_arr = np.concatenate(power_spec_arr_list, axis=0)
    power_norm_arr = np.concatenate(power_norm_arr_list, axis=0)
    major_ind = np.concatenate(major_ind_list, axis=0)
        
    # data_ind = (rec_ind[:,0] == 1) & (rec_ind[:,2] == 1)
    
    # prop_arr, contour_arr, pca_arr, power_spec_arr, power_norm_arr, rec_ind = prop_arr[data_ind], contour_arr[data_ind], pca_arr[data_ind], power_spec_arr[data_ind], power_norm_arr[data_ind], rec_ind[data_ind]
    # major_label_dict = {0:'not_invasive', 1:'invasive'}
    # major_ind_str = list()
    # for ind1 in major_ind:
    #     major_ind_str.append(major_label_dict[ind1])
        
    sub_ind, sub_inter_key = fel.cell_tokenizer(rec_ind[:,:3], return_inter = True)
    sub_inter_key_str = {key: gdi.get_rec_ind_title(value) for key, value in sub_inter_key.items()}
    sub_ind_str = list()
    for ind1 in sub_ind:
        sub_ind_str.append(sub_inter_key_str[ind1])
    
    

    
    file_name = 'ML_analysis_2-4'
    
    
    path_name = 'type_invasive_subtype_condition'
        
        
    
    feature_name = copy.deepcopy(title_list)
    feature_name.append('power_norm')
    PCA_dim_lim = 200
    power_dim_lim = 100
    for ii in range(min(pca_arr.shape[1], PCA_dim_lim)):
        feature_name.append('PCA%03d'%ii)
    
    for ii in range(min(power_spec_arr.shape[1], power_dim_lim)):
        feature_name.append('power%03d'%(ii+1))
        
    feature_arr = np.concatenate([prop_arr, power_norm_arr, pca_arr[:,:PCA_dim_lim], power_spec_arr[:,:power_dim_lim]], axis=1)
    feature_arr_norm, feature_norm_data = fel.z_score_norm_axis1(feature_arr)
        
    
        
    path_name = os.path.join('./phet_dataset', path_name)
    os.makedirs(path_name, exist_ok= True)
    
    
    subtype_file_path = os.path.join(path_name, file_name + '_types.csv')
    classes_file_path = os.path.join(path_name, file_name + '_classes.csv')
    feature_name_file_path = os.path.join(path_name, file_name + '_feature_names.csv')
    mtx_file_path = os.path.join(path_name, file_name + '_matrix.mtx')

    generated_file_paths = [
        subtype_file_path,
        classes_file_path,
        feature_name_file_path,
        mtx_file_path,
    ]
    for generated_file_path in generated_file_paths:
        if os.path.isfile(generated_file_path):
            os.remove(generated_file_path)

    subtype_dict = {'subtypes': sub_ind_str}
    save_pd_csv(subtype_file_path, subtype_dict)
    
    classes_dict = {'classes': major_ind}
    save_pd_csv(classes_file_path, classes_dict)
    
    feature_name_dict = {'features': feature_name}
    save_pd_csv(feature_name_file_path, feature_name_dict)
    
    mtx_data = feature_arr_norm[:, :]
    scipy.io.mmwrite(mtx_file_path, mtx_data)
    
    os.makedirs(os.path.join(path_name, 'result'), exist_ok=True)
