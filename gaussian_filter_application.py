from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import argrelextrema
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import time


working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob_SMALL\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs' # prep dataset (20 images)
#working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs' # full dataset (2.4k images)

sigma = 10
step = 1
median_size=5
parameter_set = (working_dir, sigma, median_size)


def plot_data(plot_values, step):
    ydata, xdata = plot_values
    x = xdata[::step]
    y = ydata[::step]
    plt.plot(x, y, '-ob', alpha=0.7, label='sigma ={} \n step = {}'.format(sigma, step))
    plt.title(r'Minimum(Intensity) as a function of the projection')
    ax = plt.gca()
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % step != 0]
    plt.legend()
    plt.xlabel('projection')
    plt.ylabel('Min. value of the intensity')
    plt.savefig('min_values_develop_version_sigma{}_median{}.png'.format(sigma, median_size), dpi=300)
    plt.show()
    return 0


def fit_sin(x_values, A, B, C, D):
    return A*np.sin(C*x_values + D) + B

def fit_poly():
    pass




def add_filter(dir, sigma, median_size):
    i = 0
    list_min_val = []
    list_filenames = []
    try:
        for filename in os.listdir(dir):
            num1 = i+1
            num2 = len(os.listdir(dir))
            progress = 100*(num1/num2)
            raw_img = cv.imread(os.path.join(dir, filename), 2)
            median_filtered_img = median_filter(raw_img, median_size)                   # 1.1 sec
            cv.imshow('median filtered image (file: {})'.format(filename), median_filtered_img)
            cv.waitKey(0)
            gaussian_filtered_img = gaussian_filter(median_filtered_img, sigma=sigma)   # 0.3 sec
            list_min_val.append(np.amin(gaussian_filtered_img))
            list_filenames.append(filename)
            if (num1 % 2 == 0):
                print('{}% done.'.format(round(progress)))
            else:
                pass
            i += 1
    except FileNotFoundError :
        print('File not fount! Check the working directory.')
    arr_min_val = np.array(list_min_val)
    arr_filename = np.array(list_filenames)
    return arr_min_val, arr_filename


plot_values = add_filter(*parameter_set)

plot_data(plot_values, step)

