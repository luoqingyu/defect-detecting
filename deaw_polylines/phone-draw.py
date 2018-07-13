# -*- coding: UTF-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import   pytesseract
import cv2
import  random
import math

from PIL import Image
from matplotlib import pyplot as plt
# 读取图像
import numpy as np
from matplotlib import pyplot as plt
import os
def get_romate_point(point,center,angle):

    new_point = []
    distence = []
    distence.append(point[0] - center[0])
    distence.append(point[1] - center[1])
    new_point.append( round(distence[0]* math.cos(angle) + distence[1] * math.sin(angle) + center[0]))
    new_point.append( round(-distence[0] * math.sin(angle) + distence[1]* math.cos(angle) + center[1]))
    return new_point
def get_romate_point(point,center,angle):
    new_point = []
    distence = []
    distence.append(point[0] - center[0])
    distence.append(point[1] - center[1])
    new_point.append( round(distence[0]* math.cos(angle) + distence[1] * math.sin(angle) + center[0]))
    new_point.append( round(-distence[0] * math.sin(angle) + distence[1]* math.cos(angle) + center[1]))
    return new_point
if __name__ == '__main__':
    img = cv2.imread('/home/lqy/Data/phone/628-chhose/2/Image__2018-06-28__11-34-40.bmp')
    #channel, w_img, h_img = img.shape[::-1]
    #print(math.sin(90*math.pi/180))
    #print(w_img,h_img)
    #print(get_romate_point((0,0),(w_img/2,h_img/2),180*math.pi/180))
    channel, w_img, h_img = img.shape[::-1]
    pts = np.array([[610, 411], [2150,412], [2150, 561], [609,561]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    #new = []
    #new_pts = []
    #print(img.shape)
    #for row in pts:
            #new_pts.append(get_romate_point(row,(w_img/2,h_img/2),180*math.pi/180))
    #new.append(new_pts)
    #print(new_pts)
    #new = np.array(new_pts)
    #new = new.reshape((-1, 1, 2))

    #new_pts =  pts.reshape((-1, 1, 2))
    #图片旋转
    #print(new_pts)
    M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), 180, 1)
    img = cv2.warpAffine(img, M, (w_img, h_img))
    #画线
    cv2.line(img, (610, 411), (2150, 412), (0, 255, 0), 5)
    
    #画四边形框
    dst = cv2.polylines(img,np.int32(pts), True, (0, 0, 255),3)

    Pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    Pts = Pts.reshape((-1, 1, 2))
    cv2.polylines(img, [Pts], True, (0, 255, 255), 33)

    Pts = np.array([[610, 411], [2150,412], [2150, 561], [609,561]], np.int32)

    new_pts = []

    print(w_img,h_img)
    for row in Pts:
        new_pts.append(get_romate_point(row, (w_img / 2, h_img / 2), 180 * math.pi / 180))
    new_pts =np.array(new_pts)


    print(new_pts)
    new_pts1 = new_pts.reshape((-1, 1, 2))


    Pts = Pts.reshape((-1, 1, 2))
    print(Pts)




    cv2.polylines(img, [Pts], True, (0, 255, 255), 3)
    cv2.polylines(img, np.int32([new_pts1]), True, (0, 255, 255), 3)






    #画矩形框
    cv2.rectangle(img, (580,351), (2193,648),(0, 0, 255), 3)
    cv2.imwrite('result.bmp', img)

    #img[351:648, 580:2193]
    #img_cropped = img[351:648, 580:2193]
    #print(get_romate_point((0, 0), (20, 50), 180 * math.pi / 180))






