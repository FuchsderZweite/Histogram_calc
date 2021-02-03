import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
import os
import cv2 as cv



class Fit:
    p0_10 = np.empty(11)
    p0_10.fill(0.1)
    degree = 10

    def __init__(self, median_kernel_size, sigma, fit_function, dir, dir_rocessed, dir_save):
        self.median_kernel_size = median_kernel_size
        self.sigma = sigma
        self.fit_function = fit_function
        self.dir = dir
        self.dir_rocessed = dir_rocessed
        self.dir_save = dir_save

    def rearrange_data(self, xdata):
        y = xdata
        single_step = 360 / xdata.size
        x = np.arange(0, 360, single_step)
        return x, y

    def get_data(self):
        list_min_val = []
        list_filenames = []
        try:
            for filename in os.listdir(self.dir):
                img_raw = cv.imread(os.path.join(self.dir, filename), 2)
                #img_median_filter = median_filter(img_raw, self.median_kernel_size)                   # 1.1 sec
                #img_gaussian_filter = gaussian_filter(img_median_filter, sigma=self.sigma)   # 0.3 sec
                #cv.imwrite(self.dir_save + '\gaussian_' + filename, img_gaussian_filter)
                #list_min_val.append(np.amin(img_gaussian_filter))
                list_min_val.append(np.amin(img_raw))
                list_filenames.append(filename)
        except FileNotFoundError:
            print('File not fount! Check the working directory.')
        arr_min_val = np.array(list_min_val)
        x, y = self.rearrange_data(arr_min_val)
        coeff, pcov = curve_fit(self.polynomial, x, y, p0=self.p0_10)
        yfit = self.polynomial(x, *coeff)
        return x, y, yfit, coeff, self.degree

    def polynomial(self, x, *coeff):
        return coeff[0]*x**10 + coeff[1]*x**9 + coeff[2]*x**8 + coeff[3]*x**7 \
               + coeff[4]*x**6 + coeff[5]*x**5 + coeff[6]*x**4 + coeff[7]*x**3\
               + coeff[8]*x**2 + coeff[9]*x + coeff[10]