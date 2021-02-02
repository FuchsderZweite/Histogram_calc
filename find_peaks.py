from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import find_peaks
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


dir_source = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob_SMALL\240_binned4x4' # prep dataset (240 images, binned 2x2)
#dir_source = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs' # full dataset (2.4k images)
#working_dir =r'C:\Users\Rechenfuchs\Documents\GitHub\dummy_data_for_hist_calc'
dir_save = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob_SMALL\processed'


sigma = 10
step = 1
median_size=5
parameter_set = (dir_source, sigma, median_size)


def find_extrema(xdata, ydata):
    result = find_peaks(ydata, prominence=1)
    return result


def add_filter(dir, sigma, median_size):
    list_min_val = []
    list_min_val_arg = []
    list_filenames = []
    print(
        'The source directory contains {} files.\n'
        'The process will take approx. {} sec.'.format(len(os.listdir(dir)), round(1.4 * len(os.listdir(dir)))))
    try:
        for filename in os.listdir(dir):
            raw_img = cv.imread(os.path.join(dir, filename), 2)
            median_filtered_img = median_filter(raw_img, median_size)                   # 1.1 sec
            gaussian_filtered_img = gaussian_filter(median_filtered_img, sigma=sigma)   # 0.3 sec
            #cv.imwrite(dir_save + '\gaussian_' + filename, gaussian_filtered_img)
            list_min_val.append(np.amin(gaussian_filtered_img))
            list_filenames.append(filename)

    except FileNotFoundError :
        print('File not fount! Check the working directory.')
    arr_min_val = np.array(list_min_val)
    arr_min_val_arg = np.array(list_min_val_arg)
    arr_filename = np.array(list_filenames)
    x, y = rearrange_data(arr_min_val, arr_filename)
    return x, y

def rearrange_data(xdata, ydata):
    y = xdata[::step]
    single_step = 360/xdata.size
    x = np.arange(0, 360, single_step)
    return x, y

x, y = add_filter(dir_source, sigma=sigma, median_size=median_size)

peaks, properties = find_extrema(x, y)