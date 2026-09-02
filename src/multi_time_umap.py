# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 09:39:59 2025

@author: peins
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle
import pandas as pd
import umap

import sys
import data_read_temp as DRT
import get_dataset_info as gdi


plt_ind = 100
# current attempt number to change analysis setting
att_num = 0

# extracted feature save position
data_save_path_dict = {0:['./data_save[0]'], 1:['./data_save[1]'], 2:['./data_save[2]'], 3:['./data_save[0, 1, 2]'], 
                       4:['./data_save[0]', './data_save[1]', './data_save[2]']}

# feature selection information
# numbers indicating feature numbers of each array
feature_sel_dict = dict()
if att_num <= 3:
    
    feature_sel_dict['prop_sel'] = np.sort([8, 7, 3, 6, 4]).astype(int)
    feature_sel_dict['prop_stand_sel'] = np.sort([]).astype(int)
    feature_sel_dict['pca_sel'] = np.sort([]).astype(int)
    feature_sel_dict['power_sel'] = np.sort([]).astype(int)
    feature_sel_dict['power_norm_sel'] = np.sort([0]).astype(int)
    
else:
    
    feature_sel_dict['prop_sel'] = np.sort([8, 7,  6,]).astype(int)
    feature_sel_dict['prop_stand_sel'] = np.sort([3,4]).astype(int)
    feature_sel_dict['pca_sel'] = np.sort([]).astype(int)
    feature_sel_dict['power_sel'] = np.sort([]).astype(int)
    feature_sel_dict['power_norm_sel'] = np.sort([0]).astype(int)

if __name__ == '__main__':
    plt.close('all')
    
    result_dict = DRT.base_data_read(data_save_path_dict[att_num], feature_sel_dict)
    prop_arr = result_dict['prop_arr'] #all region property array (raw_data)
    prop_arr_stand = result_dict['prop_arr_stand'] # regin property array 
    feature_arr = result_dict['feature_arr']
    manual_invasive = result_dict['manual_invasive']
    rec_ind = result_dict['rec_ind']
    feature_arr_norm = result_dict['feature_arr_norm']
    
    
    umap_options = {'min_dist': 0.05, 'n_neighbors': 4, 'random_state': 200}
    
    data_path1 = './%02d_data_mean1'%att_num
    os.makedirs(data_path1, exist_ok=True)
    
    data_path2 = './%02d_data_mean2'%att_num
    os.makedirs(data_path2, exist_ok=True)

    if len(np.unique(rec_traj[:,0]))>1:
        data_path3 = './%02d_data_mean3'%att_num
        os.makedirs(data_path3, exist_ok=True)
        
        data_path4 = './%02d_data_mean4'%att_num
        os.makedirs(data_path4, exist_ok=True)
    
    traj_keys = DRT.rec_to_inter_keys(rec_ind[:,:4])
    fr_step = 6
    fr_stamp = np.unique(rec_ind[:,4])
    fr_max_list = list()
    for traj_key1 in traj_keys:
        traj_ind  = DRT.rec_to_ind(rec_ind[:,:4], traj_key1)
        if any(traj_ind):
            max_fr1 = np.max(rec_ind[traj_ind, 4])
            fr_max_list.append(max_fr1)
            
        
    fr_max = np.min(fr_max_list)
    fr_lens = np.append(np.arange(fr_step + 1,fr_max - fr_step, fr_step), fr_max)
    for fr_len in fr_lens:
        for fr_st in range(0, fr_max-fr_len+1, fr_step):
            plt.close('all')
            print(fr_st, fr_len)
            data_mean = list()
            rec_traj = list()
            fr_ind = (fr_st <= rec_ind[:,4]) & (rec_ind[:,4] <= (fr_st + fr_len))
            
            for traj_key1 in traj_keys:
                traj_ind = DRT.rec_to_ind(rec_ind[:,:4], traj_key1) & fr_ind
                rec_ind_sub = rec_ind[traj_ind]
                rec_traj1 = np.concatenate([rec_ind_sub[0,:4], rec_ind_sub[[0, -1], 4]])
                data_mean1 = np.mean(feature_arr_norm[traj_ind, :], axis=0)
                data_mean.append(data_mean1)
                rec_traj.append(rec_traj1)
            
            data_mean = np.stack(data_mean, axis=0)
            rec_traj = np.stack(rec_traj, axis=0)
            
            umap_reducer = umap.UMAP(**umap_options)
            umap_traj = umap_reducer.fit_transform(data_mean)
            
            data_dict = dict()
            data_dict['name'] = 'st_%02d_len_%02d'%(fr_st, fr_len)
            data_dict['data'] = data_mean
            data_dict['rec'] = rec_traj
            data_dict['umap'] = umap_traj
            
            cell_token, inter_key_dict = DRT.cell_tokenizer(rec_traj[:,:3], return_inter= True)
            token_name_dict = {token: gdi.get_rec_ind_title(inter_key) for token, inter_key in inter_key_dict.items()}
            token_name_list = ['ex_' + gdi.get_rec_ind_title(inter_key)[3:].replace('_','\n') for token, inter_key in inter_key_dict.items()]
            
            manual_invasive_traj = DRT.get_flux_mean_fr_win ([manual_invasive], rec_traj, rec_ind)[0]
            
            DRT.draw_label_umap_marked (umap_traj, cell_token, save_path  = os.path.join(data_path1, '%03d_type_'%plt_ind + data_dict['name'] + '.png'), 
                                        label_dict = token_name_dict, legend_on=True, title = data_dict['name'])
            
            DRT.draw_label_umap_marked (umap_traj, manual_invasive_traj, save_path  = os.path.join(data_path1, '%03d_inv_'%plt_ind + data_dict['name'] + '.png'), 
                                        label_dict = {0:'Not_invasive', 1:'invasive'}, legend_on=True, title = data_dict['name'])            
            
            
            np.save(os.path.join(data_path1, '%03d_data_'%plt_ind + data_dict['name'] + '.npy'), data_dict, allow_pickle=True)
            
            DRT.draw_label_umap_marked (umap_traj, cell_token, save_path  = os.path.join(data_path2, '%03d_type_'%plt_ind + data_dict['name'] + '.png'), 
                                        label_dict = token_name_dict, legend_on=False, title = data_dict['name'])
            
            DRT.draw_label_umap_marked (umap_traj, manual_invasive_traj, save_path  = os.path.join(data_path2, '%03d_inv_'%plt_ind + data_dict['name'] + '.png'), 
                                        label_dict = {0:'Not_invasive', 1:'invasive'}, legend_on=False, title = data_dict['name'])
            
            if len(np.unique(rec_traj[:,0]))>1:
                uni_ex_num = np.unique(rec_traj[:,0])
                for ex_num1 in uni_ex_num:
                    ex_ind = rec_traj[:,0] == ex_num1
                    cell_token_sub1, inter_key_dict_sub = DRT.cell_tokenizer(rec_traj[ex_ind,:3], return_inter= True)
                    token_name_dict_sub = {token: gdi.get_rec_ind_title(inter_key) for token, inter_key in inter_key_dict_sub.items()}
                    token_name_list_sub = ['ex_' + gdi.get_rec_ind_title(inter_key)[3:].replace('_','\n') for token, inter_key in inter_key_dict_sub.items()]
                    
                    cell_token_sub2 = -np.ones(rec_traj.shape[0], dtype = int)
                    cell_token_sub2[ex_ind] = cell_token_sub1
                    
                    DRT.draw_label_umap_marked (umap_traj, cell_token_sub2, save_path  = os.path.join(data_path3, '%03d-ex%01d_type_'%(plt_ind, ex_num1) + data_dict['name'] + '.png'), 
                                                label_dict = token_name_dict_sub, legend_on=True, title = data_dict['name'])
                    
                condition_dict = {'Fibrin_DMSO':[(0,0,0), (1,0,1), (2,0,3)], 
                                  'Matrigel_DMSO':[(1,0,2)], 
                                  'Fibrin_Dinac_1uM':[(1,1,1), (2,1,3)],
                                  'Fibrin_Dinac_10uM':[(0,2,0), (2,2,3)],
                                  'Fibrin_TMZ_1uM':[(1,3,1), (2,3,3)],
                                  'Fibrin_TMZ_10uM':[(0,4,0), (2,4,3)],}
                Ex_name_dict = {0:'Ex0', 1:'Ex1', 2:'Ex2'}
                
                DRT.draw_label_umap_marked (umap_traj, rec_traj[:,0], save_path  = os.path.join(data_path4, '%03d-cond%01d_type_'%(plt_ind, 0) + data_dict['name'] + '.png'), 
                                            label_dict = Ex_name_dict, legend_on=True, title = data_dict['name'])
                
                for ii, (condition_name1, condition_list1) in enumerate(condition_dict.items()):
                    
                    cond_ind = np.stack([DRT.rec_to_ind(rec_traj, cond1) for cond1 in condition_list1], axis=1)
                    cond_ind = np.logical_or.reduce(cond_ind, axis=1)
                    
                    condition_token_sub = -np.ones(rec_traj.shape[0], dtype = int)
                    condition_token_sub[cond_ind] = rec_traj[cond_ind,0]
                    
                    Ex_name_dict2 = {key: value + '_' + condition_name1 for key, value in Ex_name_dict.items() if key in np.unique(rec_traj[cond_ind,0])}
                    
                    DRT.draw_label_umap_marked (umap_traj, condition_token_sub, save_path  = os.path.join(data_path4, '%03d-cond%01d_type_'%(plt_ind, ii+1) + data_dict['name'] + '.png'), 
                                                label_dict = Ex_name_dict2, legend_on=True, title = data_dict['name'])
                
                
            
            plt_ind += 1
            
            
            
            
    