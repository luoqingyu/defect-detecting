import matplotlib.pyplot as plt
import numpy as np
import cv2

#img = cv2.imread('/home/public/Datas/phone/iphone_data/IMG_2386.JPG',0)
img = cv2.imread('/home/public/Datas/phone/hei/IMG_2561.JPG',0)
bins = np.arange(257)

hist, bins = np.histogram(img, bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
