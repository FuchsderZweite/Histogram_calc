import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv



def gaussian(x, A, B, mu1, mu2, sig1, sig2):
    return A * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.))) + B * np.exp(
        -np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))


best_guess = [300000, 150000, 1, 1, 1, 0.1]
popt, covt = curve_fit(gaussian, xdata=test_file.bin_center, ydata=test_file.ys, p0=best_guess)
A, B, mu1, mu2, sig1, sig2 = popt


xspace = np.linspace(0, 1, 2000)
plt.bar(test_file.bin_center, test_file.ys, width=test_file.xs[1] - test_file.xs[0], color='green', label=r'')
plt.plot(xspace, gaussian(xspace, *popt), color='red', linestyle='-', linewidth=2.5, alpha=0.7, label=r'gaussian fit')
plt.title('Histogram')
plt.xlabel('bins (size: ')
plt.ylabel('count $N$')
plt.show()