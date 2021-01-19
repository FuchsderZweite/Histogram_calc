import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv


dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Nazila\2020-05-06_experiments\2020-05-06_SNR_meas_step_wedge_spectra'




def find_first_peak(mu1, mu2):
    if mu1 < mu2:
        return mu1
    else:
        return mu2


def write_to_file(x_val, file_name):
    f = open('x_values_of_peaks.txt', 'a+', encoding='utf-8')
    f.write(file_name + ',' + x_val)
    f.close()


def gaussian(x, A, B, mu1, mu2, sig1, sig2):
    return A*np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.))) + B*np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))


def extract_file_name(dir):
    return os.path.basename(dir)



def load_images_from_folder(working_dir):
    dir = working_dir + '\*\*.tif'
    images = []
    for filename in os.listdir(dir):
        img = cv.imread(os.path.join(dir,filename))
        if img is not None:
            images.append(img)
    return images


#abc = load_images_from_folder(dir)


# ===== READ IMG ======
img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2)
bins = 200
ys, xs, patches = plt.hist(img.ravel(), bins=bins)
bin_center = np.array([0.5 * (xs[i] + xs[i+1]) for i in range(len(xs)-1)])

best_guess = [1, 1, 1, 1, 1,1]
popt, covt = curve_fit(gaussian, xdata=bin_center, ydata=ys, p0=best_guess)
A, B, mu1, mu2, sig1, sig2 = popt




x_value_of_peak = find_first_peak(mu1, mu2)
#write_to_file(x_value_of_peak)




xspace = np.linspace(0, 1, 2000)
plt.bar(bin_center, ys, width=xs[1] - xs[0], color='navy', label=r'')
plt.plot(xspace, gaussian(xspace, *popt), color='red', linewidth=2.5, label=r'gaussian fit')
plt.title('Histogram')
plt.xlabel('bins (size: ')
plt.ylabel('count $N$')

#plt.hist(ys, xs)
plt.show()








'''
bins = np.arange(0, 256, 10) # fixed bin size
#plt.xlim([min(img_sorted)-5, max(img_sorted)+5])
plt.hist(img_sorted, bins=bins, alpha=0.5)
plt.title('Random Gaussian data (fixed bin size)')
plt.xlabel('variable X (bin size = 5)')
plt.ylabel('count')

plt.show()
'''