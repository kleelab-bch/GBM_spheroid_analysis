# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:03:16 2025

@author: peins
"""
import inspect
import numpy as np
import os

base_path = '../data'
dataset0_info = dict()



dataset1_info = dict()
dataset1_info['target_path'] = ['New_20250205_Data/*']
dataset1_info['target_path_rec'] = ['New_20250205_Data']
dataset1_info['condition_str'] = {'Fibrin_Ctl':(0,1), 'Matrigel_Ctl': (0,2), 'Fibrin_1uM_Dinac':(1,1), 'Fibrin_1uM_TMZ':(3,1),}
dataset1_info['condition_str_rev'] = {value:key for key, value in dataset1_info['condition_str'].items()}
dataset1_info['png_folder'] = 'png'
dataset1_info['png_format'] = '????.png'
dataset1_info['png_format_rec'] = lambda fr_num: '%04d.png'%fr_num
dataset1_info['mask_folder'] = 'mask'    
dataset1_info['spheroid_format'] = r'_\d+$'
dataset1_info['spheroid_format_rec'] = lambda sp_num: '_%01d'%sp_num
dataset1_info['exception_file'] = 'err.txt'
dataset1_info['remove_object'] = 30000
dataset1_info['remove_hole'] = 100000




dataset_info_all = { 1: dataset1_info,}



def img_mask_path_fun(rec_ind1):
    
    dataset_sel = dataset_info_all[rec_ind1[0]]
    # import pdb;pdb.set_trace()
    path_cand = [os.path.join(base_path, dataset_sel['target_path_rec'][a], 
                            dataset_sel['condition_str_rev'][(rec_ind1[1], rec_ind1[2])] + dataset_sel['spheroid_format_rec'](rec_ind1[3]),
                            dataset_sel['png_folder'], dataset_sel['png_format_rec'](rec_ind1[4])) for a in range(len(dataset_sel['target_path_rec']))]
    
    
    sel_ind = np.where([os.path.isfile(a) for a in path_cand])[0][0]
    
    png_path = os.path.join(base_path, dataset_sel['target_path_rec'][sel_ind], 
                            dataset_sel['condition_str_rev'][(rec_ind1[1], rec_ind1[2])] + dataset_sel['spheroid_format_rec'](rec_ind1[3]),
                            dataset_sel['png_folder'], dataset_sel['png_format_rec'](rec_ind1[4]))
    
    mask_path = os.path.join(base_path, dataset_sel['target_path_rec'][sel_ind], 
                            dataset_sel['condition_str_rev'][(rec_ind1[1], rec_ind1[2])] + dataset_sel['spheroid_format_rec'](rec_ind1[3]),
                            dataset_sel['mask_folder'], dataset_sel['png_format_rec'](rec_ind1[4]))
    
    remove_object = dataset_sel['remove_object'] 
    remove_hole = dataset_sel['remove_hole']
    
    title = get_rec_ind_title(rec_ind1)
    
    return png_path, mask_path, remove_object, remove_hole, title




def get_rec_ind_title(rec_ind1):
    
    title = ''
    
    if len(rec_ind1)>0:
        title = title + 'ex_%01d'%rec_ind1[0]
        dataset_sel = dataset_info_all[rec_ind1[0]]
    
    if len(rec_ind1)>2:
        title = title + '_' + dataset_sel['condition_str_rev'][(rec_ind1[1], rec_ind1[2])]
        
    if len(rec_ind1)>3:
        title = title + dataset_sel['spheroid_format_rec'](rec_ind1[3])
    
    if len(rec_ind1)>4:
        title = title + '_%03d'%rec_ind1[4]
        
        
    return title



        
        
