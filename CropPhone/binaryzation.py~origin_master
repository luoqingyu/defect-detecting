#-*- coding: UTF-8 -*-
""""
读取图片，计算灰度直方图，二值化测试
"""
import  numpy as np
import  cv2
def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256);
    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    return histImg

img =cv2.imread('/home/public/Datas/phone/iphone_data/IMG_2386.JPG',0)
#
ploter1 = cv2.namedWindow('figure1',0)
ploter2 = cv2.namedWindow('figure2',0)
ploter3 = cv2.namedWindow('figure3',0)
plot_hist = cv2.namedWindow('figure4',0)

#获取灰度直方图
hist = calcAndDrawHist(img,[255,0,0])
cv2.imshow('plot_hist',hist)
cv2.imshow('figure1',img)
GaussianBlur_img = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('figure2',GaussianBlur_img)

#最小阈值和最大阈值
canny_img = cv2.Canny(img, 20, 40)
cv2.imshow('figure3',canny_img)

if cv2.waitKey(0) == 27:     #ese  drop out
    cv2.destroyAllWindows()


