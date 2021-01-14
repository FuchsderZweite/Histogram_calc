# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import cv2





img = cv2.imread(r'C:\Users\Sergej\Desktop\abc1.tif', - 1)
max = np.amax(img)


hist = cv2.calcHist([img],[0],None,[256],[0.24,0.3])
print(bin(max))
plt.hist(img.ravel(), 256, [0.24,0.3])
plt.show()



def find_min_max(img):
    min = np.amin(img)
    max = np.amax(img)
    return min, max
