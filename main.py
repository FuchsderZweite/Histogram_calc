import numpy as np
import matplotlib.pyplot as plt
import cv2

path_to_local_img = r'/Users/sergej/Desktop/Ablage/Neuer Ordner/IMG_9998.JPG'
#path_to_local_img = 'C:\Users\Sergej\Desktop\abc1.tif'

img = cv2.imread(path_to_local_img, - 1)
max = np.amax(img)

hist = cv2.calcHist([img],[0],None,[256],[0.24,0.3])
print(bin(max))
plt.hist(img.ravel(), 256, [0.24,0.3])
plt.show()

def find_min_max(img):
    min = np.amin(img)
    max = np.amax(img)
    return min, max

def find_img_type(img):
    pass
