import numpy as np
import cv2 as cv


path = r'C:\Users\Rechenfuchs\Documents\GitHub\dummy_data_for_hist_calc'





img_raw = np.zeros([3000,1500], dtype=np.uint8)
img_raw.fill(255) # or img[:] = 255

dummy_raw = np.ones([3000,500],dtype=np.uint8)
dummy_raw.fill(255)


N = 100
i = 1

while i < N:
    noise_raw = np.random.normal(0, 255, size=(img_raw.shape[0], img_raw.shape[1]))
    noise = noise_raw.reshape(img_raw.shape[0], img_raw.shape[1]).astype('uint8')
    image = img_raw + noise
    cv.imwrite(path + '\IMG_{}.png'.format(i), image)
    i += 1



#cv.imshow('img_raw', img_raw)
#cv.imshow('img_noisy', img_raw + noise)
#cv.waitKey(0)




