from scipy.ndimage import gaussian_filter, median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt


dir_source = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob_SMALL\240_binned4x4'   #prep dataset (240 images, binned 2x2)
dir_save = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob_SMALL\processed'
dir_source = r'C:\Users\Rechenfuchs\Desktop\jakobs_data\raw'                    #dataset (240 images)

sigma = 10
step = 1
median_size=5
parameter_set = (dir_source, sigma, median_size)



def polinomial(x, *p):
    poly = 0.
    for n, A in enumerate(p):
        poly += A * x ** n
    return poly



def func_8(x, params):
    n = 8
    return params[0]*x**(n) + params[1]*x**(n-1) + params[2]*x**(n-2) + params[3]*x**(n-3) + \
           params[4]*x**(n-4) + params[5]*x**(n-5) + params[6]*x**(n-6) + params[7]*x**(n-7)+ params[n]


def func_7(x, params):
    n = 7
    return params[0]*x**(n) + params[1]*x**(n-1) + params[2]*x**(n-2) + params[3]*x**(n-3) + \
           params[4]*x**(n-4) + params[5]*x**(n-5) + params[6]*x**(n-6) + params[n]

def func_6(x, params):
    n = 6
    return params[0]*x**(n) + params[1]*x**(n-1) + params[2]*x**(n-2) + params[3]*x**(n-3) + \
           params[4]*x**(n-4) + params[5]*x**(n-5)+ params[n]

def func_5(x, params):
    n = 5
    return params[0]*x**(n) + params[1]*x**(n-1) + params[2]*x**(n-2) + params[3]*x**(n-3) + \
           params[4]*x**(n-4)+ params[n]



#def poly(x, params):
#    return params[0]*x**3 + params[1]*x**2 + params[2]*x + params[3]

#def poly3(x, params):
#    A, B, C, D = params
#    return A*x**3 + B*x**2 + C*x + D

def poly4(x, *params):
    A, B, C, D, E = params
    return A*x**4 + B*x**3 + C*x**2 + D*x + E

def poly5(x, *params):
    A, B, C, D, E, F = params
    return A*x**5 + B*x**4 + C*x**3 + D*x**2 + E*x * F

def poly6(x, *params):
    A, B, C, D, E, F, G = params
    return A*x**6 + B*x**5 + C*x**4 + D*x**3 + E*x**2 + F*x + G

def poly7(x, *params):
    A, B, C, D, E, F, G, H = params
    return A*x**7 + B*x**6 + C*x**5 + D*x**4 + E*x**3 + F*x**2 + G*x + H

def poly10(x, *params):
    A, B, C, D, E, F, G, H, I, J, K = params
    return A*x**10 + B*x**9 + C*x**8 + D*x**7 + E*x**6 + F*x**5 + G*x**4 + H*x**3 + I*x**2 + J*x + K



def add_filter(dir, sigma, median_size):
    list_min_val = []
    list_filenames = []
    print(
        'The source directory contains {} files.\n'
        'The process will take approx. {} sec.'.format(len(os.listdir(dir)), round(1.4 * len(os.listdir(dir)))))
    try:
        for filename in os.listdir(dir):
            raw_img = cv.imread(os.path.join(dir, filename), 2)
            gaussian_filtered_img = cv.imread(os.path.join(dir, filename), 2)
            #median_filtered_img = median_filter(raw_img, median_size)                   # 1.1 sec
            #gaussian_filtered_img = gaussian_filter(median_filtered_img, sigma=sigma)   # 0.3 sec
            #cv.imwrite(dir_save + '\gaussian_' + filename, gaussian_filtered_img)
            list_min_val.append(np.amin(gaussian_filtered_img))
            list_filenames.append(filename)

    except FileNotFoundError :
        print('File not fount! Check the working directory.')
    arr_min_val = np.array(list_min_val)
    x, y = rearrange_data(arr_min_val)
    return x, y


def rearrange_data(xdata):
    y = xdata[::step]
    single_step = 360 / xdata.size
    x = np.arange(0, 360, single_step)
    return x, y



x, y = add_filter(*parameter_set)

max_poly_size = np.arange(0, 16, 1)
def arbitrary_poly(x, *params):
    return sum([p*(x**i) for i, p in enumerate(params)])

yfit = np.empty()
for d in max_poly_size:
    p0 = [1] * (d + 1)
    popt, pcov = curve_fit(arbitrary_poly, x, y, p0=[1]*(d+1))
    yfit = np.append(yfit, *popt, axis=0)

    print('test')
    #yfit[d] = arbitrary_poly(x, *popt)





#p0_4 = [1.0**(-5), -7.0**(-4), 0.01, -0.1, 0.3]
#p0_5 = [7.0**(-6), 5.0**(-5), 5.0**(-4), 1.0**(-3), 1.0**(-2), 1.0**(-1)]
#p0_6 = [-1.0**(-7), -5.0**(-4), 0.001, 0.001, 0.001, 0.001, 0.1]
#p0_7 = [-1.0**(-7), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
#p0_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

#max_poly_size = np.arange(0, 16, 1)
#coeffs_ = []
#var_matrix_ = []





#for i in max_poly_size:
#    p0 = np.full(i, 0.1)
#    if p0.size < 1:
#        pass
#    else:
#        popt, pcov = curve_fit(polinomial, x, y, p0=p0)
#        print(popt)

#for j in popt:
#    coeffs_[i] = popt[j]
#    coeffs_ = np.array(coeffs_)










'''
coeffs4, var_matrix4 = curve_fit(poly4, x, y, p0=p0_4)
coeffs5, var_matrix5 = curve_fit(poly5, x, y, p0=p0_5)
coeffs6, var_matrix6 = curve_fit(poly6, x, y, p0=p0_6)
coeffs7, var_matrix7 = curve_fit(poly7, x, y, p0=p0_7)
coeffs10, var_matrix10 = curve_fit(poly10, x, y, p0=p0_10)


yfit[i] = poly4(x, *coeffs4)
yfit5 = poly5(x, *coeffs5)
yfit6 = poly6(x, *coeffs6)
yfit7 = poly7(x, *coeffs7)
yfit10 = poly10(x, *coeffs10)
yfit20 = polinomial(x, *coeffs_)
'''


color = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
linewidth = 2.5
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
