import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv


dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs'
bins=100

bin_center = 0.
bin_width = 0.
ys = np.array()



def gaussian(x, A, B, mu1, mu2, sig1, sig2):
    return A * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.))) + \
           B * np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))



def calc_histogram(dir):
    peaks = []
    j = 0
    print(type(peaks))
    for filename in os.listdir(dir):
        img = cv.imread(os.path.join(dir, filename), 2)
        ys, xs, patches = plt.hist(img.ravel(), bins=bins)
        bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
        bin_width = xs[2] - xs[1]

        x_value_peak = j
        peaks.append(x_value_peak)
        j = j + 1
    return peaks, j


def calc_histogram2(dir):
    peaks = []
    j = 0
    print(type(peaks))
    for filename in os.listdir(dir):
        img = cv.imread(os.path.join(dir, filename), 2)
        ys, xs, patches = plt.hist(img.ravel(), bins=bins)
        bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
        bin_width = xs[2] - xs[1]

        x_value_peak = j
        peaks.append(x_value_peak)
        j = j + 1
    return peaks, j


xspace = np.linspace(0, 1, 2000)
plt.bar(bin_center, ys, width=bin_width, color='green', label=r'')
plt.plot(xspace, gaussian(xspace, *popt), color='red', linestyle='-', linewidth=2.5, alpha=0.7, label=r'gaussian fit')
plt.title('Histogram')
plt.xlabel('bins (size: ')
plt.ylabel('count $N$')
plt.show()