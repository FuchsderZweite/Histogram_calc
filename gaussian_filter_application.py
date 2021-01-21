from scipy.ndimage import gaussian_filter
import cv2 as cv
import os

working_dir = r'C:\Users\Sergej\Desktop\Sample_DATA_Jakob\2020-08-13-zbl-GFK_Impact_gebogen-40kV-15W-150ms-10mit-nofilt_tifs'

sigma = 1

def add_filter(dir, sigma):
    try:
        for filename in os.listdir(dir):
            raw_img = cv.imread(os.path.join(dir, filename), 2)
            filtered_image = gaussian_filter(raw_img, sigma=sigma)

    except FileNotFoundError :
        print('File not fount! Check the working directory.')
    return filtered_image
