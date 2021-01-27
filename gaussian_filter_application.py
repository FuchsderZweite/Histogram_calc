from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import time


#working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob_SMALL\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs' # prep dataset (20 images)
#working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs' # full dataset (2.4k images)
working_dir =r'C:\Users\Rechenfuchs\Documents\GitHub\dummy_data_for_hist_calc'

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


def polynom_fit(x, A, B, C, D, E, F):
    return A*x**5 + B*x**4 + C*x**3 + D*x**2 + E*x + F




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
    return arr_min_val, arr_filename    # brauche (zum fitten) nicht die Dateinamen sondern die y-Werte!



plot_values = add_filter(*parameter_set)



ydata, xdata = plot_values
x = xdata[::step]
y = ydata[::step]

best_guess = [1, 1, 1, 1, 1, 1]
popt, covt = curve_fit(polynom_fit, xdata=x, ydata=y, p0=best_guess)
A, B, mu1, mu2, sig1, sig2 = popt


#plot_data(plot_values, step)

