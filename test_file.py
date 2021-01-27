import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv

plt.style.use('default')
#working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs'
working_dir =r'C:\Users\Rechenfuchs\Documents\GitHub\dummy_data_for_hist_calc'

bins = 100


def find_peak(bins):
    sorted_bins = bins.sort()
    p1 = sorted_bins[-1]
    p2 = sorted_bins[-2]
    # read the ys values and find the highest two bins -> best guess values.
    return


def write_to_file(x_val, file_name):
    f = open('x_values_of_peaks.txt', 'a+', encoding='utf-8')
    f.write(file_name + ',' + x_val)
    f.close()


def gaussian(x, A, B, mu1, mu2, sig1, sig2):
    return A * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.))) + B * np.exp(
        -np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))

def polynom_fit(x, A, B, C, D, E, F):
    return A*x**5 + B*x**4 +C*x**3 +D*x**2 +E*x + F





'''
def calc_histogram(dir):
    peaks = []
    for filename in os.listdir(dir):
        #img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2) # hier muss der Histogram-Rechenr rein
        img = cv.imread(os.path.join(dir, filename), 2) # hier ist ein Problem. Die Bilder werden als None deklariert -> fehler beim Einlesen
        ys, xs, patches = plt.hist(img.ravel(), bins=bins)
        bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
        bin_width = xs[1] - xs[0]

        best_guess = [300000, 150000, 1, 1, 1, 0.1]
        popt, covt = curve_fit(gaussian, xdata=bin_center, ydata=ys, p0=best_guess)
        A, B, mu1, mu2, sig1, sig2 = popt

        x_value_of_peak = find_first_peak(mu1, mu2)

        peaks = peaks.append(x_value_of_peak)

        #if img is not None:
        #   images.append(img)
    return peaks


calc_histogram(working_dir)


def extract_file_name(dir):
    head, tail = os.path.split(dir)
    return head, tail


def find_first_peak(mu1, mu2):
    if mu1 < mu2:
        return mu1
    else:
        return mu2



# ===== READ IMG ======
img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2)

ys, xs, patches = plt.hist(img.ravel(), bins=bins)
bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
bin_width = xs[1] - xs[0]


best_guess = [300000, 150000, 1, 1, 1, 0.1]
popt, covt = curve_fit(gaussian, xdata=bin_center, ydata=ys, p0=best_guess)
A, B, mu1, mu2, sig1, sig2 = popt


x_value_of_peak = find_first_peak(mu1, mu2)
#projections = extract_file_name(dir)


xspace = np.linspace(0, 1, 2000)
plt.bar(bin_center, ys, width=bin_width, color='green', label=r'')
plt.plot(xspace, gaussian(xspace, *popt), color='red', linestyle='-', linewidth=2.5, alpha=0.7, label=r'gaussian fit')
plt.title('Histogram')
plt.xlabel('bins (size: ')
plt.ylabel('count $N$')
plt.show()
'''