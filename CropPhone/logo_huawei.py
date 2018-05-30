# -*- coding: UTF-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np

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
win1 = cv2.namedWindow('xuanzhuan', flags=0)
cv2.imshow('xuanzhuan', img)
print(img)
#img = img.T
##图片尺度变换，缩放10x
height = int(img.shape[0] / 1)
weight = int(img.shape[1] / 1)
resize_img = cv2.resize(img, (weight, height), 0)
win3 = cv2.namedWindow('reshape', flags=0)
cv2.imshow('reshape', resize_img)

# 二值化图像
# ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(resize_img,115, 256, cv2.THRESH_BINARY)
win = cv2.namedWindow('two mode', flags=0)
cv2.imshow('two mode', thresh2)
sum_white = thresh2.sum(axis=0)
bianjie = []
flag = 0
print(sum_white)
for i, val in enumerate(sum_white):
    if val<17000 and flag ==0:
        bianjie.append(i)
        flag = 1

    if val> 17000 and flag == 1:
        bianjie.append(i)
        flag = 0


print(bianjie)
print(img.shape)
print(img[ :,bianjie[2]:bianjie[3]])
cut_pic = []
for bianjie_y in range(len(bianjie)):
    if bianjie_y%2==0:
        print(bianjie_y)
        cut_pic.append( thresh2[:,bianjie[bianjie_y]:bianjie[bianjie_y+1]])





for i, pic in enumerate(cut_pic):
    win3 = cv2.namedWindow(str(i), flags=0)
    cv2.imwrite('../data/huaweicut/' + str(i) + '.jpg', pic)
    cv2.imshow(str(i), pic)






if cv2.waitKey(0) == 27:  # ese  drop out
    cv2.destroyAllWindows()
