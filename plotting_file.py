import numpy as np
import matplotlib.pyplot as plt

class Plot:
    color = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
    linewidth = 2.5
    def __init__(self, xdata, ydata, yfit, coeffs, degree):
        self.xdata = xdata
        self.ydata = ydata
        self.yfit = yfit
        self.coeffs = coeffs
        self.degree = degree

    def plot(self):
        plt.scatter(self.xdata, self.ydata, marker='o', color='black', alpha=0.7, label='data')
        plt.plot(self.xdata, self.yfit, c=self.color[2], linestyle='-', linewidth=self.linewidth, alpha=0.7, label=r'$x^{}$'.format(self.degree))
        plt.savefig('min_values_.png', dpi=900)
        plt.legend()
        plt.xlabel('angle in (°)')
        plt.ylabel('Min. value of the intensity + fit')
        plt.show()
'''

plt.scatter(x, y, marker='o', color='black', alpha=0.7, label='data')
for i in max_poly_size:
    plt.plot(x, yfit[i], c=color[1], linestyle='-', linewidth=linewidth, alpha=0.7, label='$x^{}$'.format(i))

#plt.plot(x, yfit5, c=color[2], linestyle='-', linewidth=linewidth, alpha=0.7, label=r'$x^{5}$')
#plt.plot(x, yfit7, c=color[4], linestyle='-', linewidth=linewidth, alpha=0.7, label=r'$x^{7}$')
#plt.plot(x, yfit10, c=color[5], linestyle='-', linewidth=linewidth, alpha=0.7, label=r'$x^{10}$')
#plt.plot(x, yfit20, c=color[6], linestyle='-', linewidth=linewidth, alpha=0.7, label=r'$x^{20}$')

plt.savefig('min_values_sigma{}_median{}.png'.format(sigma, median_size), dpi=300)
plt.legend()
plt.xlabel('angle in (°)')
plt.ylabel('Min. value of the intensity + fits')
plt.show()

'''