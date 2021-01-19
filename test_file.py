import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv


def gaussian(x, A, B, mu1, mu2, sig1, sig2):
    return A*np.exp(-np.power(x - mu1, 2.) / (2 * np.power(sig1, 2.))) + B*np.exp(-np.power(x - mu2, 2.) / (2 * np.power(sig2, 2.)))



# ===== READ IMG ======
img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2)
bins = 100
ys, xs, patches = plt.hist(img.ravel(), bins=bins)
bin_center = np.array([0.5 * (xs[i] + xs[i+1]) for i in range(len(xs)-1)])

best_guess = [1000, 1000, 1, 1, 1,1]
popt, covt = curve_fit(gaussian, xdata=bin_center, ydata=ys, p0=best_guess)

# Generate enough x values to make the curves look smooth.
xspace = np.linspace(0, 1, 1000)
plt.bar(bin_center, ys, width=xs[1] - xs[0], color='navy', label=r'Histogram entries')
plt.plot(xspace, gaussian(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')



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