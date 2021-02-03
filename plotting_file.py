import numpy as np
import matplotlib.pyplot as plt

class Plot:
    color = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
    linewidth = 3.5

    axis_font = 14

    def __init__(self, xdata, ydata, yfit, coeffs, degree):
        self.xdata = xdata
        self.ydata = ydata
        self.yfit = yfit
        self.coeffs = coeffs
        self.degree = degree

    def plot(self):
        plt.scatter(self.xdata, self.ydata, marker='o', color='black', alpha=0.7, label='data')
        plt.plot(self.xdata, self.yfit, c=self.color[5], linestyle='-', linewidth=self.linewidth, alpha=0.7, label='$x^{}$'.format(self.degree))
        plt.legend()
        plt.tight_layout()
        plt.xlabel('angle in (°)', fontsize=self.axis_font)
        plt.ylabel('Intensity', fontsize=self.axis_font)
        plt.savefig('min_values_of_intensity.png', dpi=900)
        plt.show()

'''
for i in max_poly_size:
    plt.plot(x, yfit[i], c=color[1], linestyle='-', linewidth=linewidth, alpha=0.7, label='$x^{}$'.format(i))
'''