# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:24:30 2022

@author: peins
"""
import numpy as np
import pandas as pd
# import pickle


# Display
#from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib as mpl
#import matplotlib.cm as cm
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# import matplotlib.patches as patches
from collections import OrderedDict

# import cv2
# from skimage import filters, measure, morphology


import os
#import pickle
# import sys





save_temp = './temp'
os.makedirs(save_temp, exist_ok=True)

prop_title_dict = {0:'filled_area', 1:'major_axis_len', 2:'minor_axis_len', 3:'perimeter', 4:'convex_area', 
                   5:'eccentricity', 6:'extend', 7:'solidity', 8:'circularities', 9:'aspect_ratio', 10:'orientation',
                   11:'inten_mean', 12:'inten_std'}

prop_title_list = list(prop_title_dict.values())

attempt_num = 0

def base_data_read(data_save_path_list, feature_sel_dict):
    
    rec_ind_list = list()
    prop_arr_list = list()
    prop_arr_stand_list = list()
    contour_arr_list = list()
    pca_arr_list = list()
    power_spec_arr_list = list()
    power_norm_arr_list = list()
    manual_invasive_list = list()    
    
    for path1 in data_save_path_list:

        rec_ind = np.load(os.path.join(path1, 'rec_ind.npy'))
        prop_arr= np.load(os.path.join(path1, 'prop_arr.npy'))
        prop_arr_norm, prop_arr_norm_info = z_score_norm_axis1(prop_arr)
        contour_arr = np.load(os.path.join(path1, 'contour_arr.npy'))
        pca_arr = np.load(os.path.join(path1, 'pca_arr.npy'))
        power_spec_arr = np.load(os.path.join(path1, 'power_spec_arr.npy'))
        power_norm_arr = np.load(os.path.join(path1, 'power_norm_arr.npy'))
        
        # pca_reducer_path = os.path.join(path1, 'pca_reducer.pickle')
        
        
        manual_mask_pd = pd.read_csv(os.path.join(path1, 'manual_marking.csv'))
        manual_invasive = get_value_from_manual_masking(rec_ind, manual_mask_pd, pd_rec_ind = 'rec_ind', pd_masking = 'is_invasive')
        # manual_fluctuationg = fel.get_value_from_manual_masking(rec_ind, manual_mask_pd, pd_rec_ind = 'rec_ind', pd_masking = 'is_fluctuatiing')		
        # manual_rough = fel.get_value_from_manual_masking(rec_ind, manual_mask_pd, pd_rec_ind = 'rec_ind', pd_masking = 'is_rough')

        
        rec_ind_list.append(rec_ind)
        prop_arr_list.append(prop_arr)
        prop_arr_stand_list.append(prop_arr_norm)
        contour_arr_list.append(contour_arr)
        pca_arr_list.append(pca_arr)
        power_spec_arr_list.append(power_spec_arr)
        power_norm_arr_list.append(power_norm_arr)
        manual_invasive_list.append(manual_invasive)
        # manual_fluctuationg_list.append(manual_fluctuationg)
        # manual_rough_list.append(manual_rough)
    

    
    
    rec_ind = np.concatenate(rec_ind_list, axis=0)
    prop_arr = np.concatenate(prop_arr_list, axis=0)
    prop_arr_stand = np.concatenate(prop_arr_stand_list, axis=0)
    contour_arr = np.concatenate(contour_arr_list, axis=0)
    pca_arr = np.concatenate(pca_arr_list, axis=0)
    power_spec_arr = np.concatenate(power_spec_arr_list, axis=0)
    power_norm_arr = np.concatenate(power_norm_arr_list, axis=0)
    manual_invasive = np.concatenate(manual_invasive_list, axis=0)
    # manual_fluctuationg = np.concatenate(manual_fluctuationg_list, axis=0)
    # manual_rough = np.concatenate(manual_rough_list, axis=0)
    
    # manual_label_arr = np.concatenate([manual_invasive, manual_fluctuationg, manual_rough], axis = 1)
    
    feature_arr = np.concatenate([prop_arr[:,feature_sel_dict['prop_sel']], prop_arr_stand[:,feature_sel_dict['prop_stand_sel']], 
                                           pca_arr[:,feature_sel_dict['pca_sel']], power_spec_arr[:,feature_sel_dict['power_sel']], 
                                           power_norm_arr[:,feature_sel_dict['power_norm_sel']]], axis=1)
    feature_arr_norm, feature_norm_data = z_score_norm_axis1(feature_arr)
    result_dict = dict()
    result_dict['prop_arr'] = prop_arr
    result_dict['prop_arr_stand'] = prop_arr_stand
    result_dict['contour_arr'] = contour_arr
    result_dict['pca_arr'] = pca_arr
    result_dict['power_spec_arr'] = power_spec_arr
    result_dict['power_norm_arr'] = power_norm_arr
    result_dict['rec_ind'] = rec_ind
    result_dict['feature_arr'] = feature_arr
    result_dict['feature_arr_norm'] = feature_arr_norm
    result_dict['manual_invasive'] = manual_invasive


    return result_dict

def data_list_for_plot1():
    result_dict = base_data_read()
    
    feature_arr_dict = {0: result_dict['prop_arr'], 1: result_dict['power_norm_arr'], 2: result_dict['pca_arr'], 3: result_dict['power_spec_arr'], 4: result_dict['rec_ind'], 5:result_dict['manual_invasive']}
    feature_name_dict = {0: prop_title_list, 
                         1: ['power_spec_norm'], 
                         2: ['PCA%03d'%a for a in range(result_dict['pca_arr'].shape[1])],
                         3: ['power%03d'%(a+1) for a in range(result_dict['power_spec_arr'].shape[1])],                         
                         4: ['exp_day', 'condition', 'env_number', 'speroid_num', 'fr_num'],
                         5: ['manual_invasive']}

    
    plot_feature = np.array([[0,0,0], [0,1,0], [0,2,0], [0,3,0], [0,7,0], [0,8,0], [0,11,0], [0,12,0], [1,0,0], [5,0,0], [4,3,0], [4,4,0]])
    data_list, title_list, option_list = make_data_title_option_list(plot_feature, feature_arr_dict, feature_name_dict)
    
    return data_list, title_list, option_list


def data_list_for_plot2():
    result_dict = base_data_read()
    
    feature_arr_dict = {0: result_dict['prop_arr'], 1: result_dict['power_norm_arr'], 2: result_dict['pca_arr'], 3: result_dict['power_spec_arr'], 4: result_dict['rec_ind'], 5:result_dict['manual_invasive']}
    feature_name_dict = {0: prop_title_list, 
                         1: ['power_spec_norm'], 
                         2: ['PCA%03d'%a for a in range(result_dict['pca_arr'].shape[1])],
                         3: ['power%03d'%(a+1) for a in range(result_dict['power_spec_arr'].shape[1])],                         
                         4: ['exp_day', 'condition', 'env_number', 'speroid_num', 'fr_num'],
                         5: ['manual_invasive']}

    
    plot_feature = np.array([[0,0,0], [0,11,0], [1,0,0], [5,0,0]])
    data_list, title_list, option_list = make_data_title_option_list(plot_feature, feature_arr_dict, feature_name_dict)
    
    return data_list, title_list, option_list

# Utils
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
        ind = rec_to_ind(rec_ind, rec_ind1)
        result[ind] = pd_data1[-len_mask:]
    
    assert (all(result > -1))
    
    return result

        
def rec_to_ind(rec, key):
    if isinstance(rec, list):
        rec = np.stack(rec, axis =1)
    ind1 = np.ones(rec.shape[0], dtype = bool)
    for jj in range(len(key)):
        ind1 = ind1 & (rec[:,jj] == key[jj])
    
    return ind1

def rec_to_inter_keys(rec, sort = True):
    if isinstance(rec, list):
        rec = np.stack(rec, axis =1)
    tuple_recs = [tuple(rec[jj,:]) for jj in range(rec.shape[0]) if all(rec[jj,:] >=0)]
    inter_keys = list(OrderedDict.fromkeys(tuple_recs))
    
    if sort:
        inter_keys_arr = np.stack(inter_keys, axis=0)
        max_key = np.max(inter_keys_arr,axis=0)+1       
        
        ampli_key = np.array([np.prod(2*max_key[(ii+1):]) for ii in range(max_key.shape[0])])
        sort_key = [ np.sum(inter_keys_arr[ii,:] * ampli_key) for ii in range(len(inter_keys_arr)) ]
        sort_order = np.argsort(sort_key)
        
        inter_keys = [inter_keys[ii] for ii in sort_order.tolist()]
        # import pdb;pdb.set_trace()
        
    return inter_keys

def z_score_norm1(data):
    
    np_mean = np.mean(data.flatten())
    np_std = np.std(data.flatten())
    data_norm = (data - np_mean)/ np_std 
    
    return data_norm, np_mean, np_std

def z_score_restore1(data_norm, np_mean, np_std):
    
    data = (data_norm * np_std) + np_mean 
    
    return data

def z_score_norm_axis1(data):
    
    data_norm = np.zeros(data.shape)
    norm_info = np.zeros([2, data.shape[1]])
    for jj in range(data.shape[1]):
        data_norm1, data_mean, data_std = z_score_norm1(data[:,jj])
        data_norm[:,jj] = data_norm1
        norm_info[0,jj] = data_mean
        norm_info[1,jj] = data_std
        
    return data_norm, norm_info

def z_score_restore_axis1(data_norm, norm_info):
    data = np.zeros(data_norm.shape)
    for jj in range(data.shape[1]):
        
        data[:,jj] = z_score_restore1(data_norm[:,jj], norm_info[0,jj], norm_info[1,jj])
        
    
    return data

def make_data_title_option_list(plot_feature, feature_arr_dict, feature_name_dict):
    
    data_list = list()
    title_list = list()
    option_list = list()
    
    for plot1 in plot_feature:
        data_list.append(feature_arr_dict[plot1[0]][:, plot1[1]])
        title_list.append(feature_name_dict[plot1[0]][plot1[1]])
        option_list.append(plot1[2])
        
    
    return data_list, title_list, option_list

def cell_tokenizer(rec, return_inter = False):
    inter_keys = rec_to_inter_keys(rec, sort= True)
    cell_token_arr = -np.ones(rec.shape[0], dtype = np.int16)
    inter_dict = dict()
    for cell_token, key1 in enumerate(inter_keys):
        cell_ind = rec_to_ind(rec, key1)
        cell_token_arr[cell_ind] = cell_token
        inter_dict[cell_token] = key1
    
    if return_inter:        
        return cell_token_arr, inter_dict
    else:
        return cell_token_arr

def get_flux_mean_fr_win (data_win_list, rec_flux, rec_win):
    
    data_list_flux = list()
    for ii in range(len(data_win_list)):
        data_list_flux.append(list())
    
    for rec_flux1 in rec_flux:
        temp1 = np.ones(rec_win.shape[0], dtype= bool)
        for ii in range(rec_win.shape[1]-1):
            temp1 = temp1 & (rec_win[:,ii] == rec_flux1[ii])
        temp1 = temp1 & (rec_win[:,-1] >= rec_flux1[-2])  & (rec_win[:,-1] < rec_flux1[-1])
        for ii in range(len(data_win_list)):
            data_list_flux[ii].append(np.mean(data_win_list[ii][temp1]))
        
    
    for ii in range(len(data_win_list)):
        data_list_flux[ii] = np.array(data_list_flux[ii])
    
    
    return data_list_flux

color =  mpl.colormaps['Set3'](np.linspace(0, 1, 12))
plt_ind = 0


def mod_color(ind):
    ind = np.mod(int(ind), len(color))
    
    return color[ind]

def draw_label_umap_marked (umap_result, label, marker = None, ax = None, alpha = 1, save_path  = None, title = None, label_dict = None, legend_on = False):
    if label.ndim >1:
        label = label[:,-1]
    
    uni_label = np.unique(label)
    
    if ax is None:
        global plt_ind
        ax_None = True
        fig = plt.figure(plt_ind)
        ax = fig.add_subplot(111)
        ax.set_facecolor('gray')
        plt_ind +=1
    else:
        ax_None = False
    
    
    for label1 in uni_label.tolist():
        label_ind = label == label1
        #print(label1, np.sum(label_ind))
        if label1 < 0:
            ax.scatter(umap_result[label_ind, 0], umap_result[label_ind, 1], s=5, c = 'k', alpha = alpha, zorder = 1)
        else:
            if label_dict is None:
                label_str = 'label:%02d'%label1
            else:
                label_str = label_dict[label1]
            ax.scatter(umap_result[label_ind, 0], umap_result[label_ind, 1], s=5, c = [mod_color(label1)], alpha = alpha, label= label_str, zorder = 1)
        
    if not marker is None:
        ax.scatter(umap_result[marker, 0], umap_result[marker, 1], s=10, c = 'r', marker = 'o', alpha = 1, zorder = 2)
    
    if title is not None:
        ax.set_title(title)
    
    ax.axis('equal')
    umap_size = [np.max(umap_result[:,0])- np.min(umap_result[:,0]), np.max(umap_result[:,1])- np.min(umap_result[:,1])]
    ax.axis([np.min(umap_result[:,0])-umap_size[0]/10, np.max(umap_result[:,0])+umap_size[0]/10, np.min(umap_result[:,1])-umap_size[1]/10, np.max(umap_result[:,1])+umap_size[1]/10])
    if legend_on:
        ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi = 200)
        plt.close()
        return ax
    else:
        if ax_None:
            plt.show()
            return ax
        else:
            return ax
        

if __name__ == '__main__':
    
    print(1)

    

