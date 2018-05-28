#-*- coding: UTF-8 -*-
import cv2
import matplotlib.pyplot as plt
import  numpy as np
#1.读取灰度图像
#img = cv2.imread('/home/public/Datas/phone/iphone_data/IMG_2380.JPG', 0)  # 直接读为灰度图像
img = cv2.imread('/home/public/Datas/phone/hei/IMG_2564.JPG', 0)  # 直接读为灰度图像
#img = cv2.imread('/home/lqy/1.JPG', 0)


##图片尺度变换，缩放10x
height = int(img.shape[0]/10)
weight = int(img.shape[1]/10)
resize_img = cv2.resize(img,(weight,height),0)
win3 = cv2.namedWindow('reshape', flags=0)
cv2.imshow('reshape', resize_img)


# 二值化图像
#ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(resize_img, 110, 250,cv2.THRESH_BINARY )
win = cv2.namedWindow('two mode', flags=0)
cv2.imshow('two mode', thresh2)

#canny算子检测边缘
canny_img = cv2.Canny(thresh2, 100, 100)
canny_img1=canny_img
win1 = cv2.namedWindow('canny', flags=0)
cv2.imshow('canny', canny_img)

#### 轮廓检测
input_img = thresh2
image,contours, hierarchy = cv2.findContours(input_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours( input_img,contours,  -1,  (0, 255, 0),  3)
win3 = cv2.namedWindow('detection bianjie', flags=0)
cv2.imshow("detection bianjie", input_img)

#ret, thresh3 = cv2.threshold(img, 110, 250, cv2.THRESH_TRUNC)
#ret, thresh4 = cv2.threshold(img, 110, 250, cv2.THRESH_TOZERO)
#ret, thresh5 = cv2.threshold(img, 110, 250, cv2.THRESH_TOZERO_INV)
#titles = ['img', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#for i in range(6):
    #plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    #plt.title(titles[i])
    #plt.xticks([]), plt.yticks([])
#plt.show()





##检测线段
# input_img = canny_img
# lines = cv2.HoughLinesP(input_img, 1, np.pi / 180, 100, 100, 10)
# for x1, y1, x2, y2 in lines[0]:
#     cv2.line(input_img, (x1, y1), (x2, y2), (0, 255, 0), 100)
# win4 = cv2.namedWindow('xianduan', flags=0)
# cv2.imshow('xianduan', input_img)

###霍夫变换
#最后说明多少个点决定一条直线
input_img  = canny_img1
lines = cv2.HoughLines(input_img ,1,np.pi/10,40) #这里对最后一个参数使用了经验型的值
lines1 = lines[:,0,:]#提取为为二维
for rho,theta in lines1[:]:
    print (rho,theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    #print(x1,y1,x2,y2)
    input_img = cv2.line(input_img,(x1,y1),(x2,y2),(255,0,0),2)

win5 = cv2.namedWindow('HF', flags=0)
cv2.imshow('HF', input_img)

print (lines1)

#将直线减少到四条
print (lines)
rhos = []
rhos.append(lines1[0][0])
for  k in lines1[:]:
    biaozhi =True
    for z in rhos:
        if (abs(k[0]-z)<10):
            biaozhi =False
            print(False)
    if(biaozhi):
        rhos.append(k[0])
print  (rhos)


#求四条直线平均值
ave_line = []
for z in rhos:
    list_rhbos = []
    list_theta = []
    for k in lines1[:]:
        print(k[0])
        if (abs(k[0]-z)<10):
            list_rhbos.append(k[0])
            list_theta.append(k[1])
    print (z,list_theta,list_rhbos)
    ave_line.append ( [int( sum(list_rhbos)/len(list_rhbos)),sum(list_theta)/len(list_theta)])
print (ave_line)



#计算直线的两点
Line = []
def get_two_point_line(rho,theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return  [x1,y1],[x2,y2]


for rho,theta in ave_line[:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    Line.append([[x1,y1],[x2,y2]])


#计算直线方程
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

#计算两条直线交点
def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

#计算4条直线的交点
jiaodian = []
for k in ave_line:
    for j in ave_line:
        if abs(k[1]-j[1])>0.5:
            p1 =get_two_point_line(k[0],k[1])
            p2 = get_two_point_line(j[0],j[1])
            L1 = line(p1[0],p1[1])
            L2 = line(p2[0],p2[1])
            R = intersection(L1,L2)
            jiaodian.append(R)
#去除重复交点
jiaodian_quchong = list(set(jiaodian))
#截取分割的图片
print(jiaodian_quchong)




if cv2.waitKey(0) == 27:     #ese  drop out
    cv2.destroyAllWindows()

