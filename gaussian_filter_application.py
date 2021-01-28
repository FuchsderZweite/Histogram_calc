from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import time


working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob_SMALL\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs' # prep dataset (33 images)
#working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs' # full dataset (2.4k images)
#working_dir =r'C:\Users\Rechenfuchs\Documents\GitHub\dummy_data_for_hist_calc'

sigma = 10
step = 1
median_size=5
parameter_set = (working_dir, sigma, median_size)


def plot_data(plot_values, fit_values, step):
    y_data, x_data = plot_values
    poly3, poly4, poly5, poly6 = fit_values
    x = x_data[::step]
    y = y_data[::step]
    plt.scatter(x, y, 'ob', alpha=0.7, label='sigma ={} \n median size = {} \n step = {}'.format(sigma, median_size, step))
    plt.plot(x, fit_values[0], '-r', alpha=0.7, label='sigma ={} \n median size = {} \n step = {}'.format(sigma, median_size, step))
    plt.plot(x, fit_values[1], '-r', alpha=0.7, label='sigma ={} \n median size = {} \n step = {}'.format(sigma, median_size, step))
    plt.plot(x, fit_values[2], '-r', alpha=0.7, label='sigma ={} \n median size = {} \n step = {}'.format(sigma, median_size, step))
    plt.plot(x, fit_values[3], '-r', alpha=0.7, label='sigma ={} \n median size = {} \n step = {}'.format(sigma, median_size, step))
    plt.title(r'Minimum(Intensity) as a function of the projection')
    ax = plt.gca()
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % step != 0]
    plt.legend()
    plt.xlabel('projection')
    plt.ylabel('Min. value of the intensity')
    plt.savefig('min_values_develop_version_sigma{}_median{}.png'.format(sigma, median_size), dpi=300)
    plt.show()
    return 0


def get_peaks(plot_values):
    y_data, x_data = plot_values
    y = y_data[::step]
    x_data = np.arange(y_data.size)
    x = x_data[::step]
    #F = (np.amax(y) - np.amin(y))/ 2
    #best_guess = [1,1,1,1,1, F]
    return x,y


def polynom_fit(peak_data):
    x, y = peak_data
    range = np.arange(3,15)
    fit3 = []
    fit4 = []
    fit5 = []
    fit6 = []
    try:
        fit3 = np.polyfit(x, y, 3)
    except:
        print('No fit could be found for degree=3.')
    try:
        fit4 = np.polyfit(x, y, 4)
    except:
        print('No fit could be found for degree=4.')
    try:
        fit5 = np.polyfit(x, y, 5)
    except:
        print('No fit could be found for degree=5.')
    try:
        fit6 = np.polyfit(x, y, 6)
    except:
        print('No fit could be found for degree=6.')
    return fit3, fit4, fit5, fit6




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
            list_min_val.append(np.amin(gaussian_filtered_img))
            list_filenames.append(filename)

    except FileNotFoundError :
        print('File not fount! Check the working directory.')
    arr_min_val = np.array(list_min_val)
    arr_min_val_arg = np.array(list_min_val_arg)
    arr_filename = np.array(list_filenames)
    return arr_min_val, arr_filename

plot_values = add_filter(*parameter_set)
peaks = get_peaks(plot_values)
fit_values = polynom_fit(peaks)
#poly3, poly4, poly5, poly6 = fit_values

plot_data(plot_values, fit_values, step)

