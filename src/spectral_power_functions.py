# import argparse
import os
import fnmatch
import math
import string

import cv2
# import exifread
# import _pickle as cPickle
# import copy

import numpy as np
# import scipy as sp
from scipy import interpolate
# from scipy.stats import rankdata
# from scipy.stats import pearsonr
# from scipy.stats import f as f_distribution
# from scipy.stats import chi2 as chisq_distribution
# from scipy.stats import norm as norm_distribution
# from numpy import random as nprand
#import pandas as pd

# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.cm as mpcm
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib.lines as mlines
# from matplotlib.patches import Polygon
# import matplotlib.colors as mpcolors

# import statsmodels.api as sm

# from PIL import Image, ImageDraw

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis2d')
logger.setLevel(logging.INFO)

MAXK = 21

ANNOTS = ['Normal', 'NAT', 'Tumor']
DAYNUMS = ['0','6']
MODELS = ['zero', 'one', 'two']

MICRONS_PER_PIXEL = 0.51190476 # defined by optics
WIDTH_MICRONS = [0, 100, 200, 300, 400, 500, 600, 700]
HEIGHT_MICRONS = [0, 100, 200, 300, 400, 500 ]

def get_basenames(indir):
    basenames = []
    for file in sorted(os.listdir(indir)):
        if fnmatch.fnmatch(file, '*.txt'):
            basename = os.path.splitext(file)[0]
            basenames.append(basename)
    logger.info('files: %s', ' '.join(basenames))
    return (basenames)


def get_files_recursive(indir, pattern_str):
    ret = []
    for root, dirnames, filenames in os.walk(indir):
        for filename in fnmatch.filter(filenames, pattern_str):
            ret.append(os.path.join(root, filename))
    return (ret)


def convert_fullpath_to_dict(fullpath_list, check_tokens=True):
    base_dict = dict()
    for my_full in fullpath_list:
        (my_root, my_base) = os.path.split(my_full)
        # check that days match
        toks = my_base.split('_')

        if (check_tokens):
            errors = 0
            for t in toks:
                if ('Day' in t) and (t not in my_root):
                    # logger.warn('Day does not match: %s %s', my_root, my_base)
                    errors += 1
                elif ('NAT' in t) and (t not in my_root):
                    # logger.warn('NAT does not match: %s %s', my_root, my_base)
                    errors += 1
                elif ('Normal' in t) and (t not in my_root):
                    # logger.warn('Normal does not match: %s %s', my_root, my_base)
                    errors += 1
            if (errors > 0):
                logger.warn('skipping inconsistent file %s %s', my_root, my_base)
                continue
        if my_base in base_dict:
            logger.warn('repeated basename %s fullpath %s %s', my_base, base_dict[my_base], my_full)
            continue
        base_dict[my_base] = my_full
    return (base_dict)


def match_coord_image(all_coord, all_image):
    pairs = []
    coord_dict = convert_fullpath_to_dict(all_coord, check_tokens=False)
    image_dict = convert_fullpath_to_dict(all_image, check_tokens=False)
    image_keys = sorted(image_dict.keys())
    coord_matched = dict()
    image_matched = dict()
    for image_base in image_keys:
        coord_base = string.replace(image_base, '_K14.tif', '_xy.txt')
        if coord_base in coord_dict:
            pairs.append((coord_dict[coord_base], image_dict[image_base]))
            coord_matched[coord_dict[coord_base]] = True
            image_matched[image_dict[image_base]] = True
    return (pairs, coord_matched, image_matched)


def file_to_points(mask):
    image = cv2.imread(mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours,key = cv2.contourArea)
    points = np.squeeze(largest_contour)
    return (points)


def get_contour_length(points):
    m = len(points)
    dim = points.shape[1]
    contour_length = np.zeros(m)
    prev = 0.0
    for i in range(len(points)):
        j = (i - 1) % m
        mysq = 0.0
        for k in range(dim):
            term = points[i][k] - points[j][k]
            mysq += term * term
        delta = math.sqrt(mysq)
        contour_length[i] = prev + delta
        prev = contour_length[i]
    # logger.info('contour length %f', contour_length[m-1])
    return (contour_length)


def get_interpolate(points, contour_in, contour_out):
    # interpolate the contour to a regularly spaced grid
    nrow = len(contour_out)
    ncol = points.shape[1]
    # logger.info('interpolating to %s rows, %s cols', nrow, ncol)
    # create an array with the proper shape
    ret = np.zeros((nrow, ncol))

    # add the last point as the initial point with distance 0
    c0 = np.zeros(1)
    x = np.concatenate((c0, contour_in))
    p_last = np.array(points[[-1], :])
    new_pts = np.concatenate((p_last, points))
    # logger.info('countour %s %s', str(contour_in[0:5]), str(x[0:5]))
    # logger.info('shapes %s %s', str(points.shape), str(p_last.shape))
    # logger.info('orig %s new %s last %s', str(points[0:5,:]), str(new_pts[0:5,:]), str(p_last))

    for k in range(ncol):
        y = new_pts[:, k]
        fn = interpolate.interp1d(x, y, kind='linear')
        y_grid = fn(contour_out)
        for i in range(nrow):
            ret[i, k] = y_grid[i]

    return ret


def get_area(points):
    n = len(points)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    area = abs(area) / 2.0
    return area


def get_rfft(points_grid):
    n = points_grid.shape[0]
    dim = points_grid.shape[1]
    nfft = int(n / 2) + 1
    ret = np.zeros((nfft, dim), dtype=complex)
    for k in range(dim):
        a = np.fft.rfft(points_grid[:, k])
        # logger.info('assumed shape %s', str(ret.shape))
        # logger.info('actual shape %s', str(a.shape))
        ret[:, k] = a
    # logger.info('rfft start end:\n%s\n%s', str(ret[0:5,:]), str(ret[-5:,:]))
    return (ret)


def get_irfft(points_rfft):
    nk = points_rfft.shape[0]
    dim = points_rfft.shape[1]
    n = 2 * (nk - 1)
    ret = np.zeros((n, dim))
    for k in range(dim):
        a = np.fft.irfft(points_rfft[:, k])
        # logger.info('assumed shape %s', str(ret.shape))
        # logger.info('actual shape %s', str(a.shape))
        ret[:, k] = a
    # logger.info('irfft start end:\n%s\n%s', str(ret[0:5,:]), str(ret[-5:,:]))
    return (ret)


def get_power(points_rfft):
    nfreq = points_rfft.shape[0]
    dim = points_rfft.shape[1]
    power = np.zeros(nfreq)
    for k in range(dim):
        a = np.abs(points_rfft[:, k])  # a is a vector
        power = power + a ** 2  # element-wise accumulation
    # the first element of power should be <x>^2 + <y>^2 + ..., can set to zero
    # the last element of power should probably be doubled
    # logger.info('power: %s ... %s', str(power[0:10]), str(power[-5:]))
    return (power)


def normalize_points_by_power(xy_hat):
    nk = xy_hat.shape[0]
    dim = xy_hat.shape[1]
    n = 2 * (nk - 1)
    xy_hat_norm = xy_hat.copy()
    for k in range(dim):
        xy_hat_norm[0, k] = 0.0
    #    points_centered[:,k] = np.fft.irfft(rfft_centered[:,k])

    xy_norm = np.zeros((n, dim))

    # this normalization sets the first fourier component to that of a unit circle
    # for a circle with radius r, power = n^2 r^2 / 2 where n = number of points
    # so multiply by 1/r = n / sqrt(2 x power)
    p1 = 0
    for k in range(dim):
        p1 = p1 + np.abs(xy_hat[1, k]) ** 2
    facpower = float(n) / math.sqrt(2.0 * p1)
    xy_hat_norm = facpower * xy_hat_norm

    for k in range(dim):
        xy_norm[:, k] = np.fft.irfft(xy_hat_norm[:, k])

    return (xy_norm)


def normalize_points(points_rfft, area):
    nk = points_rfft.shape[0]
    dim = points_rfft.shape[1]
    n = 2 * (nk - 1)
    points_centered = np.zeros((n, dim))
    points_normarea = np.zeros((n, dim))
    points_normpower = np.zeros((n, dim))
    rfft_centered = points_rfft.copy()
    for k in range(dim):
        rfft_centered[0, k] = 0.0
        points_centered[:, k] = np.fft.irfft(rfft_centered[:, k])

    # for unit radius, area = pi r^2
    # so r = sqrt(area / pi)
    # divide by sqrt(area/pi) to get unit radius
    facarea = 1.0 / math.sqrt(area / math.pi)
    rfft_normarea = facarea * rfft_centered

    # this normalization sets the first fourier component to that of a unit circle
    # for a circle with radius r, power = n^2 r^2 / 2 where n = number of points
    # so multiply by 1/r = n / sqrt(2 x power)
    power = 0
    for k in range(dim):
        power = power + np.abs(points_rfft[1, k]) ** 2
    facpower = float(n) / math.sqrt(2.0 * power)
    rfft_normpower = facpower * rfft_centered

    for k in range(dim):
        points_normarea[:, k] = np.fft.irfft(rfft_normarea[:, k])
        points_normpower[:, k] = np.fft.irfft(rfft_normpower[:, k])

    power_normarea = np.zeros(nk)
    power_normpower = np.zeros(nk)
    # normalize so that a unit circle has power = 1 for first component
    # multiply by 2/n^2
    prefac = 2.0 / (n * n)
    for k in range(dim):
        ar = np.abs(rfft_normarea[:, k])
        power_normarea = power_normarea + prefac * ar ** 2
        ap = np.abs(rfft_normpower[:, k])
        power_normpower = power_normpower + prefac * ap ** 2
    return (points_centered, points_normarea, points_normpower,
            power_normarea, power_normpower)


def get_power_moments(power_rfft, n=-1):
    mysum = 0.0
    mymean = 0.0
    mylen = len(power_rfft)
    if (n == -1):
        n = len(power_rfft)
    assert (n <= mylen), 'bad length: ' + str(n)
    # logger.info('n = %s', str(n))
    mysum = sum(power_rfft[2:n])
    myfreq = range(n)
    mysum1 = sum(power_rfft[2:n] * myfreq[2:n])
    mysum2 = sum((power_rfft[2:n] * myfreq[2:n]) * myfreq[2:n])
    return (mysum, mysum1, mysum2)

def get_norm_power_from_file(file,nfft):
        xy_raw = file_to_points(file)
        contour_length = get_contour_length(xy_raw)
        # make a regular grid
        tot_length = contour_length[-1]
        ds = tot_length / float(nfft)
        contour_grid = np.linspace(ds, tot_length, num=nfft)
        xy_interp = get_interpolate(xy_raw, contour_length, contour_grid)
        area = get_area(xy_interp)
        form_factor = (tot_length * tot_length) / (4.0 * math.pi * area)
        xy_hat = get_rfft(xy_interp)
        power_hat = get_power(xy_hat)
        # points_irfft = get_irfft(points_rfft)
        # try some normalizations
        #(points_centered, points_normarea, points_normpower,
        #power_normarea, power_normpower) = normalize_points(points_rfft, area)
        #write_boundary(boundarypdf, coord_file,
        #                   points_grid, points_irfft,
        #                   points_centered, points_normarea, points_normpower,
        #                   power_normarea, power_normpower)
        power_norm = np.copy(power_hat)
        power_norm[0] = 0.0
        fac = 1.0 / power_norm[1]
        power_norm = fac * power_norm
        return(xy_raw, xy_interp, xy_hat, tot_length, area, form_factor, power_norm)


def compute_weighted_spectral_power(power_norm):
    """
    Compute the weighted spectral power based on normalized power spectrum.

    Parameters:
    power_norm (np.ndarray): Normalized power spectrum of length M/2 + 1.

    Returns:
    float: Weighted spectral power.
    """
    M = (len(power_norm) - 1) * 2  # Total number of interpolation points (e.g., 512)
    k_values = np.arange(2, len(power_norm))  # Frequency indices from 2 to M/2

    # Compute weighting factors
    sin_term = np.sin(np.pi * k_values / M)
    cos_term = np.cos(np.pi * k_values / M)
    weight = (M / np.pi) ** 2 * sin_term ** 2 * cos_term ** 2

    # Compute weighted spectral power
    weighted_power = np.sum(weight * (power_norm[k_values] / power_norm[1]))

    return weighted_power

## function added by Daehyung Kim

def mask_to_points2(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours,key = cv2.contourArea)
    points = np.squeeze(largest_contour)
    return (points)

def get_norm_power_from_mask1(mask1,nfft):
        xy_raw = mask_to_points2(mask1)
        contour_length = get_contour_length(xy_raw)
        # make a regular grid
        tot_length = contour_length[-1]
        ds = tot_length / float(nfft)
        contour_grid = np.linspace(ds, tot_length, num=nfft)
        xy_interp = get_interpolate(xy_raw, contour_length, contour_grid)
        area = get_area(xy_interp)
        form_factor = (tot_length * tot_length) / (4.0 * math.pi * area)
        xy_hat = get_rfft(xy_interp)
        power_hat = get_power(xy_hat)
        # points_irfft = get_irfft(points_rfft)
        # try some normalizations
        #(points_centered, points_normarea, points_normpower,
        #power_normarea, power_normpower) = normalize_points(points_rfft, area)
        #write_boundary(boundarypdf, coord_file,
        #                   points_grid, points_irfft,
        #                   points_centered, points_normarea, points_normpower,
        #                   power_normarea, power_normpower)
        power_norm = np.copy(power_hat)
        power_norm[0] = 0.0
        fac = 1.0 / power_norm[1]
        power_norm = fac * power_norm
        import pdb;pdb.set_trace()
        return(xy_raw, xy_interp, xy_hat, tot_length, area, form_factor, power_norm)
    
def get_norm_spectral_power_from_contour_arr(contour_arr):
    
    spectral_power_arr = np.zeros([contour_arr.shape[0],1])
    
    for ii, contour1 in enumerate(contour_arr):
    
        xy_hat = get_rfft(contour1)
        power_hat = get_power(xy_hat)
        
        power_norm = np.copy(power_hat)
        power_norm[0] = 0.0
        fac = 1.0 / power_norm[1]
        power_norm = fac * power_norm
        
        spectral_power = compute_weighted_spectral_power(power_norm)
        
        spectral_power_arr[ii,0] = spectral_power
    
    return spectral_power_arr
        
