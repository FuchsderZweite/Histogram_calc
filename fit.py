import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import cv2 as cv


class Fit:
    dir_data = r'C:\Users\Rechenfuchs\Desktop\jakobs_data\processed'  # dataset (240 images)'
    dir_processed = None  # dataset (240 images)'
    dir_save = None
    p0_10 = np.empty(11)
    p0_10.fill(0.1)
    degree = 10

    def __init__(self, dir_data=None, dir_save=None, plot_data = None):
        self.median_kernel_size = 10
        self.gaussian_kernel_size = 5
        self.fit_function = 1
        self.dir_data = dir_data
        self.dir_rocessed = None
        self.plot_data = plot_data
        self.dir_save = dir_save
        if dir_data is not None:
            self.load_data(self.dir_data)
        else:
            print('No data to evaluate. Process stopped.')
        if dir_save is not None:
            self.data_save(dir_save)
        else:
            print('No saves will be created.')

    def rearrange_data(self, xdata):
        y = xdata
        single_step = 360 / xdata.size
        x_angle = np.arange(0, 360, single_step)
        x_step = np.arange(0, x_angle.size, 1)
        arr_x = np.vstack((x_angle, x_step))
        return arr_x, y

    def get_data(self):
        list_min_val = []
        list_filenames = []
        try:
            for filename in os.listdir(self.dir_data):
                img_raw = cv.imread(os.path.join(self.dir_data, filename), 2)
                # img_median_filter = median_filter(img_raw, self.median_kernel_size)                   # 1.1 sec
                # img_gaussian_filter = gaussian_filter(img_median_filter, sigma=self.sigma)   # 0.3 sec
                # cv.imwrite(self.dir_save + '\gaussian_' + filename, img_gaussian_filter)
                # list_min_val.append(np.amin(img_gaussian_filter))
                # negative Werte verwerfen?!
                list_min_val.append(np.amin(img_raw))
                list_filenames.append(filename)
        except FileNotFoundError:
            print('File not fount! Check the working directory.')
        arr_min_val = np.array(list_min_val)
        self.x, self.y = self.rearrange_data(arr_min_val)                         # y is 1D, but x is 2D (1D for angle and 1D for numbering
        coeff, pcov = curve_fit(self.polynomial, self.x[0], self.y, p0=self.p0_10)
        self.yfit = self.polynomial(self.x[0], *coeff)
        self.xminima, self.xmaxima, self.samples_major, self.samples_minor = self.get_min_max(self.x, self.yfit)
        self.plot()
        return self.xminima, self.xmaxima, self.samples_major, self.samples_minor

    def get_min_max(self, x, yfit):
        n = 5
        arr_maxima = argrelextrema(yfit, np.greater)
        arr_minima = argrelextrema(yfit, np.less)
        arr_extrema = np.concatenate((arr_minima, arr_maxima), axis=None)
        arr_extrema = np.sort(arr_extrema)
        intersept_major = []
        for i in arr_extrema:
            indx = x[0][i]
            intersept_major.append(indx)
        arr_inter_major = np.array(intersept_major)


        # gehe durch beide Listen und finde zum ersten Extrema x0 den darauffolgenden x1 (unabhaengig von min oder max)
        # ziehe diese Extrema von einander ab und teile den Abstand in aequidistante Abstaende
        intersept_minor = []
        #n = np.arange(arr_inter_major[i + 1], arr_inter_major[i], n)
        for i in range(len(arr_inter_major)-1):
            indx = abs(arr_inter_major[i+1] - arr_inter_major[i])/n
            abc = np.arange(arr_inter_major[i+1], arr_inter_major[i]+1, n)
            intersept_minor.append(n*(arr_inter_major[i] + abc))
        arr_inter_minor = np.array(intersept_minor)
        return arr_minima, arr_maxima, arr_inter_major, arr_inter_minor

    def polynomial(self, x, *coeff):
        return coeff[0] * x ** 10 + coeff[1] * x ** 9 + coeff[2] * x ** 8 + coeff[3] * x ** 7 \
               + coeff[4] * x ** 6 + coeff[5] * x ** 5 + coeff[6] * x ** 4 + coeff[7] * x ** 3 \
               + coeff[8] * x ** 2 + coeff[9] * x + coeff[10]

    def sampling_points(self):
        min = self.minima

    def load_data(self, dir_data):
        self.get_data()

    def data_save(self, dir_save):
        pass

    def log_file(self):
        pass

    def plot(self):
        size = 256, 16
        dpi = 900
        figsize = size[0] / float(dpi), size[1] / float(dpi)
        color = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
        linewidth = 3.5
        axis_font = 14
        plt.scatter(self.x[0], self.y, marker='o', color='black', alpha=0.7, label='data')
        plt.plot(self.x[0], self.yfit, c=color[5], linestyle='-', linewidth=linewidth, alpha=0.7, label='$x^{}$'.format(self.degree))
        plt.vlines(x=self.samples_major, ymin=0, ymax=figsize[0], colors='gray', ls='-', alpha=0.6, lw=5, label='found minima')
        plt.vlines(x=self.samples_major, ymin=0, ymax=figsize[0], colors='gray', ls='-', alpha=0.6, lw=5, label='found maxima')
        plt.vlines(x=self.samples_minor, ymin=0, ymax=figsize[0], colors='gray', ls='--', alpha=0.4,lw=2, label='equally spaced sample points')
        plt.legend()
        plt.tight_layout()
        plt.xlabel('angle in (°)', fontsize=axis_font)
        plt.ylabel('Intensity', fontsize=axis_font)
        plt.savefig('min_values_of_intensity.png', dpi=900)
        plt.show()
        return True
