import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv



def find_min_max(img):
    min = np.amin(img)
    max = np.amax(img)
    print(min)
    print(max)
    print()
    return min, max




img = cv.imread(r'C:\Users\Sergej\Desktop\abc2.tif', 2)

histSize = 256
histRange = (0, 1)
accumulate = False


hist = cv.calcHist(img, [0], None, [histSize], histRange, accumulate=accumulate)

hist_list = [val[0] for val in hist]

#Generate a list of indices
indices = list(range(0, 256))

#Descending sort-by-key with histogram value as key
hist_sorted = [(x,y) for y,x in sorted(zip(hist_list,indices), reverse=True)]

index_of_highest_peak = hist_sorted[0][0]

#Index of second highest peak in histogram
index_of_second_highest_peak = hist_sorted[1][0]

print(index_of_highest_peak)
print(index_of_second_highest_peak)


plt.hist(img.ravel(),256,histRange); plt.show()


















'''


    
def draw_image_histogram(image, channels, color='k'):
    hist = cv.calcHist([image], channels, None, [256], [0, 256])
    plt.plot(hist)
    plt.xlim([0, 256])



gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
histogram = cv.calcHist([gray_image], [0], None, [256], [0, 256])
plt.plot(histogram, color='k')
plt.show()
'''


