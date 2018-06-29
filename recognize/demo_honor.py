# -*- coding: UTF-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import   pytesseract
# 1.读取灰度图像
# img = cv2.imread('/home/public/Datas/phone/iphone_data/IMG_2380.JPG', 0)  # 直接读为灰度图像
img = cv2.imread('/home/public/Datas/phone/shenzhen/1.jpg', 0)  # 直接读为灰度图像
# img = cv2.imread('/home/lqy/1.JPG', 0)
print(img.shape)
center = (img.shape[0]/2,img.shape[1]/2)
print(center)
rows,cols = img.shape[:2]
#第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
#M = cv2.getRotationMatrix2D((rows/2,cols/2),90,1)
#img = img.T
def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))
img = img.swapaxes(1,0)
img = fz(img)
#第三个参数：变换后的图像大小
#img = cv2.warpAffine(img,M,(rows,rows))

print(img)
#img = img.T
##图片尺度变换，缩放10x
height = int(img.shape[0] / 1)
weight = int(img.shape[1] / 1)
resize_img = cv2.resize(img, (weight, height), 0)


# 二值化图像
# ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(resize_img,50, 250, cv2.THRESH_BINARY)

code = pytesseract.image_to_string(thresh2, lang='chi_sim')
print(code)


if cv2.waitKey(0) == 27:  # ese  drop out
    cv2.destroyAllWindows()

