import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
import os
import cv2 as cv

class Fit:
    def __init__(self, median_kernel_size, sigma, fit_function, dir, dir_save):
        self.median_kernel_size = median_kernel_size
        self.sigma = sigma
        self.fit_function = fit_function
        self.dir = dir
        self.dir_save = dir_save

    def rearrange_data(self, xdata):
        y = xdata
        single_step = 360 / xdata.size
        x = np.arange(0, 360, single_step)
        return x, y

    def add_filter(self):
        list_min_val = []
        list_filenames = []
        try:
            for filename in os.listdir(self.dir):
                img_raw = cv.imread(os.path.join(self.dir, filename), 2)
                img_median_filter = median_filter(img_raw, self.median_kernel_size)                   # 1.1 sec
                img_gaussian_filter = gaussian_filter(img_median_filter, sigma=self.sigma)   # 0.3 sec
                cv.imwrite(self.dir_save + '\gaussian_' + filename, img_gaussian_filter)
                list_min_val.append(np.amin(img_gaussian_filter))
                list_filenames.append(filename)
        except FileNotFoundError:
            print('File not fount! Check the working directory.')
        arr_min_val = np.array(list_min_val)
        x, y = self.rearrange_data(arr_min_val)
        return x, y

    def polynomial(self, x, *params):
        return A*x**10 + B*x**9 + C*x**8 + D*x**7 + E*x**6 + F*x**5 + G*x**4 H*x**3 I*x**2 + J*x + K

    po_10 = []
    coeffs4, var_matrix4 = curve_fit(polynomial, add_filter.x, add_filter.y, p0=p0_10)

class Print:
    def __init__(self, age, name):
        self.name = name
        self.age = age

    def introduce_yourself(self):
        print('His name is: ' + self.name)
        print('His age is: {}'.format(self.age))