import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv
import fit







# ===== READ IMG ======
img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2)

hist_num = 256
hist_range = [0, 1.0]
accumulate = False
'''
xs = np.arange(hist_num)
hist= cv.calcHist(img, [0], None, [hist_num], histRange, accumulate=accumulate)
ys = hist[:,0]




# ===== FIT =====
#popt, pcov = curve_fit(gaussian, xs, ys)
#y_line = gaussian(xs, *popt)




# ===== PLOT ======
plt.hist(hist, histRange, color='green')
#plt.plot(xs, y_line, '--', color='red', label="gaussian fit")
#plt.legend(loc='upper right')
plt.show()
'''


# 3.) Generate exponential and gaussian data and histograms.
#data = np.random.exponential(scale=2.0, size=100000)
#data2 = np.random.normal(loc=3.0, scale=0.3, size=15000)
data = cv.calcHist(img, [0], None, [hist_num], hist_range, accumulate=accumulate)
ys = data[:,0]
bins = np.linspace(0, 256, hist_num)
data_entries, bins = np.histogram(ys, bins=bins)


# 4.) Add histograms of exponential and gaussian data.
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])


# ==== FIT FUNC ========
def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - ((x - mean) / standard_deviation) ** 2)


# 5.) Fit the function to the histogram data.
popt, pcov = curve_fit(gaussian, xdata=binscenters, ydata=data_entries)
print(popt)
print(pcov)

# 6.)
# Generate enough x values to make the curves look smooth.
xmin, xmax = plt.xlim()
xspace = np.linspace(xmin, xmax, 5000)

# Plot the histogram and the fitted function.
plt.bar(binscenters, data_entries, width=bins[1] - bins[0], color='green', label=r'Histogram entries')
plt.plot(xspace, gaussian(xspace, *popt), color='red', linewidth=2.5, label=r'Fitted function')

# Make the plot nicer.
plt.xlim(0,hist_num)
plt.xlabel(r'bins')
plt.ylabel(r'counts N')
plt.title(r'Test plot')
plt.legend(loc='best')
plt.show()
plt.clf()








