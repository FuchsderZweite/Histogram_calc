import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import os
import cv2 as cv


class Fit:
    dir_data = r'C:\Users\Rechenfuchs\Desktop\jakobs_data\processed'  # dataset (240 images)'
    dir_processed = None  # dataset (240 images)'
    dir_save = None
    p0_10 = np.empty(11)
    p0_10.fill(0.1)
    degree = 10

    def __init__(self, dir_data=None, dir_save=None):
        self.median_kernel_size = 10
        self.gaussian_kernel_size = 5
        self.fit_function = 1
        self.dir_data = dir_data
        self.dir_rocessed = None
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
        x = np.arange(0, 360, single_step)
        return x, y

    def get_data(self, dir_data):
        list_min_val = []
        list_filenames = []
        try:
            for filename in os.listdir(dir_data):
                img_raw = cv.imread(os.path.join(dir_data, filename), 2)
                # img_median_filter = median_filter(img_raw, self.median_kernel_size)                   # 1.1 sec
                # img_gaussian_filter = gaussian_filter(img_median_filter, sigma=self.sigma)   # 0.3 sec
                # cv.imwrite(self.dir_save + '\gaussian_' + filename, img_gaussian_filter)
                # list_min_val.append(np.amin(img_gaussian_filter))
                list_min_val.append(np.amin(img_raw))
                list_filenames.append(filename)
        except FileNotFoundError:
            print('File not fount! Check the working directory.')
        arr_min_val = np.array(list_min_val)
        x, y = self.rearrange_data(arr_min_val)
        coeff, pcov = curve_fit(self.polynomial, x, y, p0=self.p0_10)
        yfit = self.polynomial(x, *coeff)
        xminima, xmaxima, intersteps = self.get_min_max(x, yfit)

        return xminima, xmaxima

    def get_min_max(self,x_angle, yfit):
        n = 5
        arr_maxima = argrelextrema(yfit, np.greater)  # (array([1, 3, 6]),)
        arr_minima = argrelextrema(yfit, np.less)  # (array([2, 5, 7]),)
        for i in arr_maxima, arr_minima:
           x_minima = x_angle[i]
           x_maxima = x_angle[i]
        # Was wenn maxima und minima unterschiedliche Anzahl haben (Also wie zu 99% der Faelle)
        n_minima = len(minima)
        n_maxima = len(maxima)

        # gehe durch beide Listen und finde zum ersten Extrema x0 den darauffolgenden x1 (unabhaengig von min oder max)
        # ziehe diese Extrema von einander ab und teile den Abstand in aequidistante Abstaende
        intersteps = abs(maxima[0] - minima[0])/n
        return minima, maxima, intersteps

    def polynomial(self, x, *coeff):
        return coeff[0] * x ** 10 + coeff[1] * x ** 9 + coeff[2] * x ** 8 + coeff[3] * x ** 7 \
               + coeff[4] * x ** 6 + coeff[5] * x ** 5 + coeff[6] * x ** 4 + coeff[7] * x ** 3 \
               + coeff[8] * x ** 2 + coeff[9] * x + coeff[10]

    def sampling_points(self):
        min = self.minima

    def load_data(self, dir_data):
        self.get_data(dir_data)

    def data_save(self, dir_save):
        pass

    def log_file(self):
        pass
