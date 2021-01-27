import numpy as np
import cv2 as cv


path = r'C:\Users\Rechenfuchs\Documents\GitHub\dummy_data_for_hist_calc'





'''

img_raw = np.zeros([3000,1500], dtype=np.uint8)
img_raw.fill(255) # or img[:] = 255
#img_gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)

#dummy_raw = np.zeros([3000,q],dtype=np.uint8)




N = 100
i = 1
q = 1

min_num = 0
max_num = 10





noise_raw = np.random.uniform(min_num, max_num, size=(img_raw.shape[0], img_raw.shape[1]))
noise = q/N * noise_raw.reshape(img_raw.shape[0], img_raw.shape[1]).astype('uint8')

img_noise = img_raw * noise
cv.imshow('img_raw', img_raw)
cv.imshow('img_noisy', img_noise)
cv.waitKey(0)




while i < N:
    if i < 25:
        noise_raw = np.random.uniform(min_num, max_num-i*2, size=(img_raw.shape[0], img_raw.shape[1]))
        noise = q/N * noise_raw.reshape(img_raw.shape[0], img_raw.shape[1]).astype('uint8')
        q += 1
    elif 25<i and i < 50:
        noise_raw = np.random.normal(min_num, max_num, size=(img_raw.shape[0], img_raw.shape[1]))
        noise = q/N * noise_raw.reshape(img_raw.shape[0], img_raw.shape[1]).astype('uint8')
        q -= 1
    elif 50 < i and i < 75:
        noise_raw = np.random.normal(min_num, max_num, size=(img_raw.shape[0], img_raw.shape[1]))
        noise = q/N * noise_raw.reshape(img_raw.shape[0], img_raw.shape[1]).astype('uint8')
        q += 1
    else:
        noise_raw = np.random.normal(min_num, max_num, size=(img_raw.shape[0], img_raw.shape[1]))
        noise = q/N * noise_raw.reshape(img_raw.shape[0], img_raw.shape[1]).astype('uint8')
        q -= 1
    img_noise = img_raw * noise
    cv.imwrite(path + '\IMG_{}.png'.format(i), img_noise)
    i += 1

#cv.imshow('img_raw', img_raw)
#cv.imshow('img_noisy', img_raw + noise)
#cv.waitKey(0)

'''


