import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv

plt.style.use('default')
#working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs'
working_dir =r'C:\Users\Rechenfuchs\Documents\GitHub\dummy_data_for_hist_calc'

bins = 100




def write_to_file(x_val, file_name):
    f = open('x_values_of_peaks.txt', 'a+', encoding='utf-8')
    f.write(file_name + ',' + x_val)
    f.close()


def gaussian(x, A, B, mu1, mu2, sig1, sig2):
    return A * np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.))) + B * np.exp(
        -np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))




a = np.arange(1,17)
x = np.array(a)
y = np.array([12,8,5,5,6,6,7,8,8,7,6,7,9,6,4,1])

def poly3(x, params):
    A, B, C, D = params
    return A*x**3 + B*x**2 + C*x + D

def poly4(x, params):
    A, B, C, D, E = params
    return A*x**4 + B*x**3 + C*x**2 + D*x + E

def poly7(x, params):
    A, B, C, D, E, F, G, H = params
    return A*x**7 + B*x**6 + C*x**5 + D*x**4 + E*x**3 + F*x**2 + G*x + H

params3 = np.polyfit(x, y, 3)
params4 = np.polyfit(x, y, 4)
params7 = np.polyfit(x, y, 7)


plt.scatter(x, y, marker='o', alpha=0.7, label=r'data')
plt.plot(x, poly3(x, params3), color='red', linestyle='-', linewidth=2.5, alpha=0.7, label=r'$x^{3}$')
plt.plot(x, poly4(x, params4), color='green', linestyle='-', linewidth=2.5, alpha=0.7, label=r'$x^{4}$')
plt.plot(x, poly7(x, params7), color='purple', linestyle='-', linewidth=2.5, alpha=0.7, label=r'$x^{7}$')
plt.title('Fit comparison')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()





#def calc_histogram(dir):
#    peaks = []
#    for filename in os.listdir(dir):
        #img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2) # hier muss der Histogram-Rechenr rein
#        img = cv.imread(os.path.join(dir, filename), 2) # hier ist ein Problem. Die Bilder werden als None deklariert -> fehler beim Einlesen
#        ys, xs, patches = plt.hist(img.ravel(), bins=bins)
#        bin_center = np.array([0.5 * (xs[i] + xs[i + 1]) for i in range(len(xs) - 1)])
#        bin_width = xs[1] - xs[0]

#        best_guess = [300000, 150000, 1, 1, 1, 0.1]
#        popt, covt = curve_fit(gaussian, xdata=bin_center, ydata=ys, p0=best_guess)
#        A, B, mu1, mu2, sig1, sig2 = popt

#        x_value_of_peak = find_first_peak(mu1, mu2)

#        peaks = peaks.append(x_value_of_peak)

        #if img is not None:
        #   images.append(img)
#    return peaks

'''
calc_histogram(working_dir)


def extract_file_name(dir):
    head, tail = os.path.split(dir)
    return head, tail



# ===== READ IMG ======


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