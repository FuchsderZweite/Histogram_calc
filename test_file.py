import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv

plt.style.use('default')
working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs'


bins = 100


print(len(os.listdir(working_dir)))

def gaussian1(x, A, mean, sigma1):
    return A * np.exp(-np.power(x - mean, 2.) / (2 * np.power(sigma1, 2.)))

def gaussian2(x, A, B, mean1, mean2, sigma1, sigma2):
    return A * np.exp(-np.power(x - mean1, 2.) / (2 * np.power(sigma1, 2.))) + \
           B * np.exp(-np.power(x - mean2, 2.) / (2 * np.power(sigma2, 2.)))



'''
def best_guess(ys, xs):
    # find index of ys value
    ys_index_1 = np.where(ys == np.amax(ys))
    xs_index_1= np.where(xs == ys_index_1)

    # find x_values for peaks guess positions
    #sorted_peaks = val.sort()
    # 1 übergebe die Histogramwerte ys -> y_values
    # 2 Sortiere ys und weise diese Werten zu
    #p1 = sorted_peaks[-1]
    #p2 = sorted_peaks[-2]
    return xs_index, x2, A, B
'''



def calc_histogram(dir):
    peaks = []
    tuning_factor = 25
    x_value_peak = 0.
    for filename in os.listdir(dir):
        print(filename)
        img = cv.imread(os.path.join(dir, filename), 2)
        x_val_arr = img.ravel()
        ys, xs, patches = plt.hist(x_val_arr, bins=bins)
        bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
        bin_width = xs[2] - xs[1]

        median_mean_diff = abs(np.mean(x_val_arr) - np.median(x_val_arr))
        #
        if median_mean_diff < bin_width*tuning_factor: # single gaussian
            best_guess = [300000, 0.2, 0.1]
            try:
                popt, covt = curve_fit(gaussian1, xdata=bin_center, ydata=ys, p0=best_guess)
                A, mean1, sigma1 = popt
                x_value_peak = mean1
            except Exception:
                pass
        else: # double gaussian
            best_guess = [300000, 150000, 1, 1, 1, 0.1]
            try:
                popt, covt = curve_fit(gaussian2, xdata=bin_center, ydata=ys, p0=best_guess)
                A, B, mean1, mean2, sigma1, sigma2 = popt
                x_value_peak = find_first_peak(mean1, mean2)
            except Exception:
                pass
        peaks.append(x_value_peak)
    print(len(peaks))
    return peaks

peaks = calc_histogram(working_dir)

xspace = np.linspace(0, 1, 2000)

x_values = [[i for i in range(len(os.listdir(working_dir)))] for j in range(len(os.listdir(working_dir)))]
plt.plot(x_values, peaks, color='red', linestyle='-', linewidth=2.5, alpha=0.7, label=r'gaussian fit')
plt.title('Histogram')
plt.xlabel('bins (size: ')
plt.ylabel('count $N$')
plt.show()




def extract_file_name(dir):
    head, tail = os.path.split(dir)
    return head, tail


def find_first_peak(mu1, mu2):
    if mu1 < mu2:
        return mu1
    else:
        return mu2



# ===== READ IMG ======

#hist_vals = np.array(ys, xs)
#bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
#bin_width = xs[1] - xs[0]


#best_guess = [300000, 150000, 1, 1, 1, 0.1]
#popt, covt = curve_fit(gaussian, xdata=bin_center, ydata=ys, p0=best_guess)
#A, B, mu1, mu2, sig1, sig2 = popt


#x_value_of_peak = find_first_peak(mu1, mu2)
#projections = extract_file_name(dir)




peaks, bin_center, bin_width, ys, *popt = calc_histogram(working_dir)

#xspace = np.linspace(0, 1, 2000)
#plt.bar(bin_center, ys, width=bin_width, color='green', label=r'')
#plt.plot(xspace, gaussian(xspace, *popt), color='red', linestyle='-', linewidth=2.5, alpha=0.7, label=r'gaussian fit')
#plt.title('Histogram')
#plt.xlabel('bins (size: ')
#plt.ylabel('count $N$')
#plt.show()