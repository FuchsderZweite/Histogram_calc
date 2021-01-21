import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv



def gaussian(x, A, B, mean1, mean2, sigma1, sigma2):
    return A * np.exp(-np.power(x - mean1, 2.) / (2 * np.power(sigma1, 2.))) + \
           B * np.exp(-np.power(x - mean2, 2.) / (2 * np.power(sigma2, 2.)))


bins = 256
img_path = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs\GFK_gebogen-40kv_0540.tif'



def read_img(dir):
    if os.path.exists(dir):
        img = cv.imread(img_path, 2)
        return img
    else:
        print('could not find file!')


img = read_img(img_path)

ys, xs, patches = plt.hist(img.ravel(), bins=bins)


#hist_vals = np.vstack((ys, xs))


#hist_vals = np.array(ys, xs)
bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
bin_width = xs[1] - xs[0]


best_guess = [300000, 150000, 1, 1, 1, 0.1]
popt, covt = curve_fit(gaussian, xdata=bin_center, ydata=ys, p0=best_guess)
A, B, mean1, mean2, sigma1, sigma2 = popt


xspace = np.linspace(0, 1, 2000)
plt.bar(bin_center, ys, width=bin_width, color='green', label=r'')
plt.plot(xspace, gaussian(xspace, *popt), color='red', linestyle='-', linewidth=2.5, alpha=0.7, label=r'gaussian fit')
plt.title('Histogram')
plt.xlabel('bins (size: ')
plt.ylabel('count $N$')
plt.show()



#x_value_of_peak = find_first_peak(mu1, mu2)
#projections = extract_file_name(dir)