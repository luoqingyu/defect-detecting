#cording:utf-8
#163
import cv2
import matplotlib.pyplot as plt
import  numpy as np
#1.读取灰度图像
#img = cv2.imread('/home/public/Datas/phone/iphone_data/IMG_2380.JPG', 0)  # 直接读为灰度图像
img = cv2.imread('../thresholding/demo.bmp', 0)  # 直接读为灰度图像
# 二值化图像
#ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 60, 255,cv2.THRESH_BINARY )
thresh3 = thresh2.copy()
win = cv2.namedWindow('two mode', flags=0)
cv2.imshow('two mode', thresh2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#腐蚀图像
eroded = cv2.erode(thresh2,kernel)
cv2.imshow("Eroded Image",eroded)
#显示差值
chazhi = thresh2 -eroded
cv2.imshow('cha1',chazhi)
#处理差值
chazhi[chazhi>0]=1
chazhi = 1-chazhi

print(chazhi)
print(thresh3.sum())

result = thresh3 * chazhi
print(result.sum())
print(np.max(chazhi))
cv2.imshow('result',result)
# #显示腐蚀后的图像
# cv2.imshow("Eroded Image",eroded)




kerne2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# #膨胀图像
dilated = cv2.dilate(thresh2,kerne2)
# #显示膨胀后的图像
cv2.imshow("Dilated Image",dilated);
chazhi2 = dilated -eroded
cv2.imshow('cha2',chazhi2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
