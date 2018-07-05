# -*- coding: UTF-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import   pytesseract
import cv2
import  random
from PIL import Image
from matplotlib import pyplot as plt
# 读取图像
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


def mkdir(path):  # 判断是否存在指定文件夹，不存在则创建
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print('mkdir'+path)
        return True
    else:
        return False
# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    child_file_name=[]
    full_child_file_list = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        #print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        full_child_file_list.append(child)
        child_file_name.append(allDir)
    return  full_child_file_list,child_file_name
def creat_img(im_name,angle, Scale,num_img,result_dir):
    print(im_name)
    img = cv2.imread(im_name, 0)
    w_img, h_img = img.shape[::-1]
    for i in range(num_img):
        romote_angle = random.uniform( -angle, angle)
        change_Scale = random.uniform(1-Scale/1.,1+Scale/1.)
        # Scale_img = cv2.resize(img2,change_Scale)
        M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2), romote_angle, change_Scale)
        dst = cv2.warpAffine(img, M, (w_img, h_img))
        result_dir =  result_dir
        img_result_file = result_dir + im_name.split('/')[-1].replace('.','') + '_' + str(romote_angle).replace('.', '_') + '_' + str(
            change_Scale).replace('.', '_') + '.bmp'
        cv.imwrite(img_result_file, dst)



if __name__ == '__main__':
    first_dir = '/home/lqy/Data/phone/628-chhose/'
    second_full_dir,second_child_dir = eachFile(first_dir)
    angle = 5                   #(-5,+5)
    Scale = 1                   #(-Scale/10,Scale/10)
    num_img =  100              #creat how many image
    result_dir = '/home/lqy/Data/phone/628-data-add'+str(angle)+'_'+str(Scale)+'/'
    mkdir(result_dir)
    for dir in second_full_dir:
        third_full_dir,third_child_dir = eachFile(dir+'/')
        x_dir = result_dir+dir.split('/')[-1]
        mkdir(x_dir)
        for  im_name in third_full_dir:
            creat_img(im_name,angle,Scale,num_img,x_dir+'/')




