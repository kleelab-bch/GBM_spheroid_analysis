# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:55:32 2024

@author: peins
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import os
import pandas as pd
from collections import OrderedDict
from sklearn.decomposition import PCA
from skimage import draw, transform, measure

import spectral_power_functions as spf
import get_dataset_info as gdi

save_temp = 'R:/'
os.makedirs(save_temp, exist_ok=True)
plt_ind = 1000
sigma_color = mpl.colormaps['bwr'](np.linspace(0,1,5))
sigma_color[2] = [0, 0, 0, 1]
num_process_max = 8

# feature extraction function

def change_contour_st_old(contour):
    ind_st = np.argmax(contour[:,0])
    ind_arr = np.concatenate([np.arange(ind_st,contour.shape[0]), np.arange(0,ind_st)], axis=0)    
    contour_shift = contour[ind_arr,:]
    
    contour_mean = np.mean(contour, axis=0)
    contour_shift = contour_shift - contour_mean
    
    return contour_shift

def change_contour_st(contour):    
    
    y_axis_close_ind = np.where(np.abs(contour[:,1]) < 1.5)[0]
    # import pdb;pdb.set_trace()
    largest_y_ind = np.argmax(contour[y_axis_close_ind,0])
    
    ind_st = y_axis_close_ind[largest_y_ind]
    ind_arr = np.concatenate([np.arange(ind_st,contour.shape[0]), np.arange(0,ind_st)], axis=0)    
    contour_shift = contour[ind_arr,:]
    
    
    return contour_shift

def fourier_contour_feature_pca(contour_all, dim = 5):
    
    
    
    xf = np.fft.fft(contour_all[:,:,1], axis = -1)
    yf = np.fft.fft(contour_all[:,:,0], axis = -1)
    fourier_feature = np.concatenate([np.real(xf), np.imag(xf), np.real(yf), np.imag(yf)], axis=1)
    
    
    reducer = PCA(n_components = dim)
    result = reducer.fit_transform(fourier_feature)
    print(reducer.explained_variance_ratio_, sum(reducer.explained_variance_ratio_))
    
    return result, reducer

def restore_pca_contour_feature(pca_value, reducer):
    if pca_value.ndim ==1:
        pca_value = pca_value[np.newaxis, :]
        
    restored_fourier = reducer.inverse_transform(pca_value)    
    
    
    fourier2 = restored_fourier.reshape([restored_fourier.shape[0], 4, -1])
    xf = fourier2[:,0] + fourier2[:,1] * 1j
    yf = fourier2[:,2] + fourier2[:,3] * 1j
    contour_x = np.real(np.fft.ifft(xf, axis=-1))
    contour_y = np.real(np.fft.ifft(yf, axis=-1))
    restored_contour = np.stack([contour_y, contour_x], axis=-1)

    return restored_contour

def power_spectrum_feature(contour_all):
    
    xf = np.fft.fft(contour_all[:,:,1], axis = -1)
    yf = np.fft.fft(contour_all[:,:,0], axis = -1)
    
    power_spec_arr_ori = np.abs(xf)**2 + np.abs(yf)**2
    boundary = int(np.round(contour_all.shape[1]/2))
    power_spec_arr = power_spec_arr_ori[:, 1:boundary]
    
    return power_spec_arr

def binary_mask_to_reduced_contour(binary_mask, contour_dim = None, major_rot = True):
    
    mask1 = binary_mask
    contour = measure.find_contours(mask1, 0.5) #contour는 axis 0: y , axis 1: x 방향이다
    
    contour_len = [a.shape[0] for a in contour]
    sel_ind = np.argmax(contour_len)
    contour = contour[sel_ind]
    mask_prop = measure.regionprops(mask1.astype('uint8'))
    mask_prop = mask_prop[0]
    
    if major_rot:

        orientation = mask_prop.orientation #이 radian 각도만큼 거꾸로 돌리면 y axis: major x_axis: minor 가 된다. 
        rot_mat = np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
        contour_rot = np.matmul(contour - mask_prop.centroid, rot_mat)
        result = change_contour_st(contour_rot)
        # result = result - mask_prop.centroid
        
    else:
        result = contour

    
    if contour_dim is not None:
        result = reduce_contour(result, dim = contour_dim)
    
    return result

def binary_mask_to_reduced_contour_spf(binary_mask, contour_dim = None, major_rot = True):
    
    mask_prop = measure.regionprops(binary_mask.astype('uint8'))
    mask_prop = mask_prop[0]
    contour = measure.find_contours(binary_mask, 0.5) #contour는 axis 0: y , axis 1: x 방향이다
    
    contour_len = [a.shape[0] for a in contour]
    sel_ind = np.argmax(contour_len)
    contour = contour[sel_ind]
    
    if major_rot:

        orientation = mask_prop.orientation #이 radian 각도만큼 거꾸로 돌리면 y axis: major x_axis: minor 가 된다. 
        rot_mat = np.array([[np.cos(orientation), -np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
        contour_rot = np.matmul(contour - mask_prop.centroid, rot_mat)
        result = change_contour_st(contour_rot)
        # result = result - mask_prop.centroid
        
    else:
        result = contour
        
    if contour_dim is not None:
        
        contour_length = spf.get_contour_length(result)
        tot_length = contour_length[-1]
        ds = tot_length / float(contour_dim)
        contour_grid = np.linspace(ds, tot_length, num=contour_dim)
        # contour_rot_shift_sim = spf.get_interpolate(contour_rot_shift, contour_length, contour_grid)
        result = spf.get_interpolate(result, contour_length, contour_grid)
    
    return result


def get_region_prop(mask1):
    
    mask1 = mask1 > 0.5
    props = measure.regionprops(mask1.astype('uint8'))[0]
    
    
    # ['filled_area', 'axis_major_length', 'axis_minor_length', 'perimeter', 'convex_area', 
    # 'eccentricity', 'extent', 'solidity', 'circularity', 'aspect_ratio',]
    prop_arr = np.zeros(10)
    try:
        prop_arr[0] = props.filled_area
        prop_arr[1] = props.major_axis_length
        prop_arr[2] = props.minor_axis_length
        prop_arr[3] = props.perimeter
        prop_arr[4] = props.convex_area
        prop_arr[5] = props.eccentricity
        prop_arr[6] = props.extent
        prop_arr[7] = props.solidity
        circularities = (4 * np.pi * props.filled_area) / (props.perimeter **2)
        prop_arr[8] = circularities
        bindingbox = props.bbox
        aspect_ratio = (bindingbox[3]-bindingbox[1])/(bindingbox[2]-bindingbox[0])
        prop_arr[9] = aspect_ratio
    except:
        import pdb;pdb.set_trace()
    
    return prop_arr

def get_region_prop2(mask1, png1):
    
    mask1 = mask1 > 0.5
    props = measure.regionprops(mask1.astype('uint8'), intensity_image= png1)[0]
    
    
    # ['filled_area', 'axis_major_length', 'axis_minor_length', 'perimeter', 'convex_area', 
    # 'eccentricity', 'extent', 'solidity', 'circularity', 'aspect_ratio',]
    prop_arr = np.zeros(13)
    try:
        prop_arr[0] = props.filled_area
        prop_arr[1] = props.major_axis_length
        prop_arr[2] = props.minor_axis_length
        prop_arr[3] = props.perimeter
        prop_arr[4] = props.convex_area
        prop_arr[5] = props.eccentricity
        prop_arr[6] = props.extent
        prop_arr[7] = props.solidity
        circularities = (4 * np.pi * props.filled_area) / (props.perimeter **2)
        prop_arr[8] = circularities
        bindingbox = props.bbox
        aspect_ratio = (bindingbox[3]-bindingbox[1])/(bindingbox[2]-bindingbox[0])
        prop_arr[9] = aspect_ratio
        prop_arr[10] = props.orientation
        # prop_arr[11] = props.intensity_mean
        
        inside_img = png1[mask1]
        back_range = 100
        background_range_x = [max(0, bindingbox[1]-back_range), min(png1.shape[1], bindingbox[3]+back_range)]
        background_range_y = [max(0, bindingbox[0]-back_range), min(png1.shape[0], bindingbox[2]+back_range)]
        #print([bindingbox[0], bindingbox[2], bindingbox[1], bindingbox[3]])
        #print([background_range_x[0], background_range_x[1], background_range_y[0], background_range_y[1]])
        
        outside_img = png1[background_range_y[0]:background_range_y[1], background_range_x[0]:background_range_x[1]]
        outside_label = mask1[background_range_y[0]:background_range_y[1], background_range_x[0]:background_range_x[1]]
        outside_mean = np.mean(outside_img[outside_label == 0])
        
        inside_mean = np.mean(inside_img) - outside_mean
        inside_std = np.std(inside_img)
        
        prop_arr[11] = inside_mean
        prop_arr[12] = inside_std
        
    except:
        import pdb;pdb.set_trace()
    
    return prop_arr

def get_region_prop_mul(mask_arr):
    
    prop_arr = np.zeros((mask_arr.shape[0], 10))
    
    for ii in range(mask_arr.shape[0]):
        prop_vec = get_region_prop(mask_arr[ii])
        prop_arr[ii] = prop_vec
    
    return prop_arr

def get_contour_data_mul(mask_arr, contour_dim, major_axis_rot = True):
    n_data = mask_arr.shape[0]
    contour_arr = np.zeros([n_data, contour_dim, 2])
    for ii in range(n_data):
        contour1 = binary_mask_to_reduced_contour(mask_arr[ii], contour_dim = contour_dim, major_rot = major_axis_rot)  
        contour_arr[ii] = contour1
        
    return contour_arr


#Utility

def contiguous_numbers(num_arr, step = 1):
    diff_arr = np.diff(num_arr, n = 1)
    cont_arr = diff_arr == step
    idx_list = contiguous_regions(cont_arr)
    idx_list2 = [[a[0], a[1]+1] for a in idx_list]
    
    return idx_list2

def contiguous_regions(condition): #https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    
    return idx

def png_list_to_mv(png_list, movie_name = None, del_png = True, fps = 5):
    if movie_name is None:
        video_name = 'R:/temp.avi'
    else:
        video_name = movie_name
        
    images = png_list
    #images = [img for img in os.listdir(save_path) if img.endswith(".png")]
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc(*'h264')
    
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))

    for image in images:
        video.write(cv2.imread(image))

    # cv2.destroyAllWindows()
    video.release()  
    
    if del_png:
        
        for del_png1 in png_list:
            os.remove(del_png1)

def to_int_list(a):
    if a is None:
        a = list()
    else:
        if isinstance(a, np.ndarray):
            a = a.tolist()
        elif isinstance(a, tuple):
            a = list(a)
        
        elif isinstance(a, float) or isinstance(a, int):
            a = [a]
            
        for ii in range(len(a)):
            a[ii] = int(a[ii])
    
    return a

def reduce_contour(contour, dim = 128):
    
    redu_ind = np.linspace(0, contour.shape[0]-1, num = dim, endpoint= True)
    redu_ind = np.round(redu_ind, decimals=0).astype(int)
    
    # import pdb;pdb.set_trace()
    
    return contour[redu_ind, :]

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
        

def cal_z_score_zero_line(np_mean, np_std):
    return -np_mean/np_std


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

def draw_label_prediction(True_label, Pred_label, ticks = ['MCF', 'mda'], save_path = None):
    global plt_ind
    
    True_label = True_label.astype(int)
    uni_label = np.unique(True_label)
    max_label = len(np.unique(True_label))
    Pred_label_round = np.round(Pred_label).astype(int)
    img_conf = np.zeros((max_label, max_label), dtype = int)
    
    for ii in range(len(True_label)):
        True_ind = uni_label == True_label[ii]
        Pred_ind = uni_label == Pred_label_round[ii]
        
        img_conf[Pred_ind, True_ind] +=1
        
    fig = plt.figure(plt_ind, figsize=(2+max_label, 2+max_label))
    plt_ind +=1
    ax = fig.add_subplot(111)
    im = ax.imshow(img_conf, vmin = 0, vmax = np.quantile(img_conf.flatten(), 0.9), cmap = 'Blues')
    for ind1 in range(img_conf.shape[0]):
        for ind2 in range(img_conf.shape[1]):
            ax.text(ind2, ind1, str(img_conf[ind1, ind2]), va = 'center', ha = 'center', size= 'x-large', c= 'r')
            
    plt.xticks(np.arange(max_label), ticks)
    plt.yticks(np.arange(max_label), ticks)
    
    plt.xlabel('True')
    plt.ylabel('Pred')
    
    temp = plt.axis()
    plt.axis([temp[0], temp[1], temp[3], temp[2]])
    
    if save_path is not None:
        plt.savefig(save_path , dpi = 200)
        plt.close()
    else:
        plt.show()

    
    return img_conf

def draw_feature_matrix(data_list, title_list, label_arr, label_name = None, save_path = None):
    
    label_arr = label_arr.astype(int)
    uni_label = np.unique(label_arr)
    uni_label = uni_label[uni_label>=0]
    
    if label_name is None:
        label_name = [str(a) for a in uni_label]
    
    data_list_norm = list()
    norm = mpl.colors.Normalize(vmin= -1.1, vmax= +1.1)
    
    for data1 in data_list:
        data1_norm, np_mean, np_std = z_score_norm1(data1)
        data_list_norm.append(data1_norm)
    
    label_ind_dict = dict()
    for label1 in uni_label:
        label_ind_dict[label1] = label_arr ==label1
    
    
    data_arr = np.zeros([len(data_list), len(uni_label)])
    
    for ii, data1 in enumerate(data_list_norm):
        for jj, label1 in enumerate(uni_label):
            data_arr[ii,jj] = np.mean(data1[label_ind_dict[label1]])
    
    
    global plt_ind
    
    fig = plt.figure(plt_ind)
    plt_ind +=1
    ax = fig.add_subplot(111)
    im = ax.imshow(data_arr, norm = norm, cmap = 'bwr')
    
    ax.set_xticks(np.arange(len(uni_label)), label_name)
    ax.set_yticks(np.arange(len(data_list)), title_list)
    
    cbar = fig.colorbar(im, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['-1 S.D.', '0', '+1 S.D.'])
    
    if save_path is not None:
        plt.savefig(save_path , dpi = 200)
        plt.close()
    else:
        plt.show()
    
        
        
    

def size_adjust(dst_range_list, src_range_list, src_lim_list): 
    #고정된 이미지 안에 다른 이미지의 일부를 잘라 넣는 과정에서 그 일부가 이미지를 벗어났을때의 보정을 위한 코드
    #format, [[dim1_min, dim1_max], [dim2_min, dim2_max]]
    
    dst_range_list2 = list()
    src_range_list2 = list()
    dim_lim = min(len(dst_range_list), len(src_range_list), len(src_lim_list))
    
    for dim1 in range(dim_lim):
        dst_range_list[dim1] = [int(a) for a in dst_range_list[dim1]]
        src_range_list[dim1] = [int(a) for a in src_range_list[dim1]]
        src_lim_list[dim1] = [int(a) for a in src_lim_list[dim1]]
        if src_range_list[dim1][0] < src_lim_list[dim1][0]:
            adj_min = int(src_lim_list[dim1][0] - src_range_list[dim1][0])
        else:
            adj_min = int(0)
        
        if src_range_list[dim1][1] > src_lim_list[dim1][1]:
            adj_max = int(src_range_list[dim1][1] - src_lim_list[dim1][1])
        else:
            adj_max = int(0)
     
        src_range1 = [src_range_list[dim1][0] + adj_min, src_range_list[dim1][1] - adj_max]
        dst_range1 = [dst_range_list[dim1][0] + adj_min, dst_range_list[dim1][1] - adj_max]
        
        src_range_list2.append(src_range1)
        dst_range_list2.append(dst_range1)
    
    return dst_range_list2, src_range_list2


def label_change(label_arr_src, change_list):
    
    uni_label_src = np.unique(label_arr_src)
    
    change_list_src = [a[0] for a in change_list]
    change_list_dst = [a[1] for a in change_list]
    
    match_check_src = [a not in change_list_src for a in uni_label_src]
    if any(match_check_src):
        match_check_src2 = uni_label_src[match_check_src]
        for label1 in match_check_src2:
            change_list.append((label1, label1))
            
    
    label_arr_dst = - np.ones(label_arr_src.shape[0], dtype = int)
    for label_src1, label_tar1 in change_list:        
        ind = label_arr_src == label_src1
        if any(ind):
            label_arr_dst[ind] = label_tar1
    
    
    return label_arr_dst
    
    


# PCA test plot functions
def sigma_plot(pca_data, pca_reducer, pca_axis = np.arange(5), save_path = './temp/'):
    global plt_ind
    
    os.makedirs(save_path, exist_ok=True)
    
    pca_mean = np.mean(pca_data, axis=0)
    pca_std = np.std(pca_data, axis=0)
    
    
    sigma_range = [-2, -1, 0, 1, 2]
    exp_ratio = pca_reducer.explained_variance_ratio_
    
    for sel_axis in pca_axis:
        pca_fake = np.stack([pca_mean]*5, axis=0)
        for ii, jj in enumerate(sigma_range):            
            pca_fake[ii,sel_axis] = pca_fake[ii,sel_axis] + pca_std[sel_axis] * jj
        
        fake_restore = restore_pca_contour_feature(pca_fake, pca_reducer)
        
        fig = plt.figure(plt_ind)
        plt_ind += 1
        ax = fig.add_subplot(111)
        
        for ii, jj in enumerate(sigma_range):
            
            plt.plot(fake_restore[ii,:,1], fake_restore[ii,:,0], c = sigma_color[ii], label='%01d sigma'%jj)
            plt.plot(fake_restore[ii,0,1], fake_restore[ii,0,0], c = sigma_color[ii], marker = '^')
            plt.plot(fake_restore[ii,-1,1], fake_restore[ii,-1,0], c = sigma_color[ii],  marker = 'v')
        plt.title('PCA_axis: %01d'%sel_axis)
        plt.xlabel('Exp_ratio: %0.4f'%exp_ratio[sel_axis])
        plt.axis('equal')
            
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(save_path, 'PCA_sigma_%02d.png'%sel_axis), dpi = 200)
        plt.close()
        
def recon_test1(contour_data, pca_data, pca_reducer, dim_lim = 5, index = np.arange(0, 500, 50), save_path = './temp/'):
    global plt_ind
    
    os.makedirs(save_path, exist_ok=True)
    
    exp_ratio = np.sum(pca_reducer.explained_variance_ratio_[:dim_lim])
    temp = np.copy(pca_data)
    temp[:,dim_lim:] =0
    contour_restore = restore_pca_contour_feature(temp, pca_reducer)
    
    for ii in index:
        plt.figure(plt_ind)        
        plt_ind +=1
        plt.plot(contour_data[ii,:,1], contour_data[ii,:,0], 'b')
        plt.plot(contour_data[ii,0,1], contour_data[ii,0,0], 'b^')
        plt.plot(contour_data[ii,-1,1], contour_data[ii,-1,0], 'bv')
        plt.plot(contour_restore[ii,:,1], contour_restore[ii,:,0], 'r')
        plt.plot(contour_restore[ii,0,1], contour_restore[ii,0,0], 'r^')
        plt.plot(contour_restore[ii,-1,1], contour_restore[ii,-1,0], 'rv')
        plt.axis('equal')
        plt.title('num_data %03d'%(ii))
        plt.xlabel('#_of_pca_dim:%02d, exp_ratio:%0.3f'%(dim_lim, exp_ratio))
        plt.savefig(os.path.join(save_path, 'PCA_recon_%02d.png'%ii), dpi = 200)
        plt.close()
    

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


def sp_csv_gen(rec_ind, save_path = 'R:/temp.csv'):
    uni_sp_ind = rec_to_inter_keys(rec_ind[:,:4])
    rec_ind_list = list()
    sp_name = list()
    for uni_sp1 in uni_sp_ind:        
        rec_ind_list.append(uni_sp1)
        sp_name.append(gdi.get_rec_ind_title(uni_sp1))
    
    temp = {'rec_ind%01d'%ii: [a[ii] for a in rec_ind_list] for ii in range(len(uni_sp1))}
    temp['sp_name'] = sp_name
    temp_pd = pd.DataFrame(data=temp)
    temp_pd.to_csv(save_path)
    # import pdb;pdb.set_trace
    

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

if __name__ == '__main__':
    print(2)
    
    
    arr_size = 500
    mask_arr = np.zeros([arr_size, 600, 600], dtype = bool)
    rand_number = np.random.uniform(low = 0, high = 1, size = [arr_size, 4])
    
    for ii in range(arr_size):
        mask = np.zeros((600, 600), dtype = float)
        rr1, cc1 = draw.ellipse(300, 300, 10 + 230*rand_number[ii,0], 10 + 230*(1-rand_number[ii,0]))
        rr2, cc2 = draw.ellipse(300, 300, 10 + 230*rand_number[ii,1], 10 + 230*(1-rand_number[ii,1]))
        
        mask[rr1, cc1] = 1
        mask = transform.rotate(mask, angle=90 * rand_number[ii,2], order=0)
        
        mask[rr2, cc2] = 1
        mask = transform.rotate(mask, angle=90 * rand_number[ii,3], order=0)
        
        mask_arr[ii] = mask>0.5
    
    prop_arr = get_region_prop_mul(mask_arr)    
    # PCA_feature_arr, PCA_feature_reducer = get_PCA_feature_mul(mask_arr, contour_dim = 512, PCA_dim = 200, major_axis_rot = True)
    contour_arr = get_contour_data_mul(mask_arr, contour_dim=512, major_axis_rot= True)
    pca_feature_arr, pca_feature_reducer = fourier_contour_feature_pca(contour_arr, dim = 200)
    
    sigma_plot(pca_feature_arr, pca_feature_reducer, pca_axis = np.arange(8), save_path = './temp/')
    recon_test1(contour_arr, pca_feature_arr, pca_feature_reducer, dim_lim = 8, index = np.arange(0, 500, 50), save_path = './temp/')
    