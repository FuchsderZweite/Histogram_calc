import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2 as cv

num_of_sin = np.arange(6)
coeff_A = np.array([1,2,3,4,5,6,7,8,9])
coeff_B = np.array([3,1,6,1,1,1,6,8,9])
coeff_C = np.array([3,1,6,1,1,1,6,8,9])
coeff_D = np.array([3,1,6,1,1,1,6,8,9])
coeffs = np.row_stack((coeff_A, coeff_B, coeff_C, coeff_D))

print(coeffs)
print(coeffs.shape)



def gaussian(x, A, B, mean1, mean2, sigma1, sigma2):
    return A * np.exp(-np.power(x - mean1, 2.) / (2 * np.power(sigma1, 2.))) + \
           B * np.exp(-np.power(x - mean2, 2.) / (2 * np.power(sigma2, 2.)))


def sum_of_sin(x, coeffs):
    sum_of_sin = 0.0
    for i in coeffs.size:
        sum_of_sin = coeffs[0:i] * np.sin(coeffs[1:i] * x + coeffs[2:i]) + coeffs[3:i]
    return sum_of_sin

