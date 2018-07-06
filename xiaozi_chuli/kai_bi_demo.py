#cording:utf-8
#163
import cv2
import matplotlib.pyplot as plt
import  numpy as np
#1.读取灰度图像
#img = cv2.imread('/home/public/Datas/phone/iphone_data/IMG_2380.JPG', 0)  # 直接读为灰度图像
img = cv2.imread('./data/xiaozi.JPG', 0)  # 直接读为灰度图像
# 二值化图像
#ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 163, 250,cv2.THRESH_BINARY )
win = cv2.namedWindow('two mode', flags=0)
cv2.imshow('two mode', thresh2)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 10))
#腐蚀图像
eroded = cv2.erode(thresh2,kernel)
#显示腐蚀后的图像
cv2.imshow("Eroded Image",eroded);
kerne2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#膨胀图像
dilated = cv2.dilate(eroded,kernel)
#显示膨胀后的图像
cv2.imshow("Dilated Image",dilated);

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
