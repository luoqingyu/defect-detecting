#-*- coding: UTF-8 -*-
import cv2
import matplotlib.pyplot as plt
import  numpy as np
#1.读取灰度图像
#img = cv2.imread('/home/public/Datas/phone/iphone_data/IMG_2380.JPG', 0)  # 直接读为灰度图像
img = cv2.imread('/home/lqy/python_scripy/defect-detecting/canny/data/demo.bmp', 0)  # 直接读为灰度图像
# 二值化图像
#ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 116, 250,cv2.THRESH_BINARY )
win = cv2.namedWindow('two mode', flags=0)
cv2.imshow('two mode', thresh2)
#canny算子检测边缘
canny_img = cv2.Canny(thresh2, 100, 100)
canny_img1=canny_img
win1 = cv2.namedWindow('canny', flags=0)
cv2.imshow('canny', canny_img)

#if cv2.waitKey(0) == 27:     #ese  drop out
#    cv2.destroyAllWindows()



gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
