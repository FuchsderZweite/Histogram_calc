import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv


def gaussian(x1, x2, mu1, mu2, sig1, sig2):
    return np.exp(-np.power(x1 - mu1, 2.) / (2 * np.power(sig1, 2.))) + np.exp(-np.power(x2 - mu2, 2.) / (2 * np.power(sig2, 2.)))



# ===== READ IMG ======
img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2)
bins = 100
ys, xs, patches = plt.hist(img.ravel(), bins=bins)




plt.hist(ys, xs)
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