#-*- coding: UTF-8 -*-
"""""
输入图片路径
输出图片的四个顶点坐标
"""""


import cv2
import  numpy as np

def process_img(img_path,mix_erzhi = 110,max_erzhi = 250,huofu_point_num=40 ,show_process_img = True):
    """
     If the coordinate of points exceeds the image region,
     then truncate it and make sure it inside the image region

     Args:
         img_path：The path of the img
         show_process_img: if ture,show the  img process
     Returns:
          lines1:return all lines  hofu detect

     """

    #1.读取灰度图像
    img = cv2.imread(img_path, 0)  # 直接读为灰度图像
    #img = cv2.imread('/home/lqy/1.JPG', 0)
    #2.图片尺度变换，缩放10x
    height = int(img.shape[0]/10)
    weight = int(img.shape[1]/10)
    resize_img = cv2.resize(img,(weight,height),0)
    # 二值化图像
    #ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(resize_img, mix_erzhi, max_erzhi,cv2.THRESH_BINARY )
    #canny算子检测边缘
    canny_img = cv2.Canny(thresh2, 100, 100)
    ###霍夫变换
    #最后说明多少个点决定一条直线
    input_img  = canny_img.copy()
    lines = cv2.HoughLines(input_img ,1,np.pi/60,huofu_point_num) #这里对最后一个参数使用了经验型的值
    lines1 = lines[:,0,:]#提取为为二维
    for rho,theta in lines1[:]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        input_img = cv2.line(input_img,(x1,y1),(x2,y2),(255,0,0),2)

    if show_process_img:
        win3 = cv2.namedWindow('reshape', flags=0)
        cv2.imshow('reshape', resize_img)

        win = cv2.namedWindow('two mode', flags=0)
        cv2.imshow('two mode', thresh2)

        win1 = cv2.namedWindow('canny', flags=0)
        cv2.imshow('canny', canny_img)

        win5 = cv2.namedWindow('HF', flags=0)
        cv2.imshow('HF', input_img)
    return  lines1

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
def get_point(lines1):
    #将直线减少到四条
    rhos = []
    rhos.append(lines1[0][0])
    for  k in lines1[:]:
        biaozhi =True
        for z in rhos:
            if (abs(k[0]-z)<30):
                biaozhi =False
                #print(False)
        if(biaozhi):
            rhos.append(k[0])



    #求四条直线平均值
    ave_line = []
    for z in rhos:
        list_rhbos = []
        list_theta = []
        for k in lines1[:]:
            #print(k[0])
            if (abs(k[0]-z)<30):
                list_rhbos.append(k[0])
                list_theta.append(k[1])
        print (z,list_theta,list_rhbos)
        ave_line.append ( [int( sum(list_rhbos)/len(list_rhbos)),sum(list_theta)/len(list_theta)])

    # 计算4条直线的交点
    jiaodian = []
    for k in ave_line:
        for j in ave_line:
            if abs(k[1] - j[1]) > 0.5:
                p1 = get_two_point_line(k[0], k[1])
                p2 = get_two_point_line(j[0], j[1])
                L1 = line(p1[0], p1[1])
                L2 = line(p2[0], p2[1])
                R = intersection(L1, L2)
                jiaodian.append(R)
    # 去除重复交点
    jiaodian_quchong = list(set(jiaodian))
    return  jiaodian_quchong





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




def main():
    img_path = '/home/public/Datas/phone/hei/IMG_2567.JPG'
    line1 = process_img(img_path = img_path )     #处理图片
    jiaodian_quchong = get_point(line1)      #获取交点
    print(jiaodian_quchong)
    if cv2.waitKey(0) == 27:     #ese  drop out
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
