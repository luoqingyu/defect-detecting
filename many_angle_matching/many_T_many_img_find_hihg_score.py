# -*- coding: UTF-8 -*-
'''
利用多角度模板对待测图片进程匹配
'''
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy  as np


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    child_file_name = []
    full_child_file_list = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        # print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        full_child_file_list.append(child)
        child_file_name.append(allDir)
    return full_child_file_list, child_file_name


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
        print('mkdir' + path)
        return True
    else:
        return False

'''
一张图片和多个模板匹配计算最高匹配分数
template_dir:模板所在文件夹
img_path：图片所在路径
result_dir：匹配结果
'''
def get_highesr_scores(template_dir, img_path, result_dir):
    img = cv.imread(img_path, 0)
    #创建保存分数和角度信息的矩阵
    scores = np.zeros((2, 80))
    full_dir, dir = eachFile(template_dir)
    #分数和模板名称的字典
    dict_angle_img_name = {}
    #
    dict_angle_img_result = {}

    for num, template_img in enumerate(full_dir):
        template = cv.imread(template_img, 0)
        methods = ['cv.TM_CCOEFF_NORMED']
        for meth in methods:
            method = eval(meth)
            # Apply template Matching
            res = cv.matchTemplate(img, template, method)
            # print(res)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
                #获取匹配分数
                scores[0][num] = max_val

            # print(template_img.split('/')[-1].split('_')[-2])
            #获取该模板的角度
            scores[1][num] = template_img.split('_')[3]
            #分数和模板名称形成字典
            dict_angle_img_name[max_val] = template_img
            #分数和匹配结果形成字典
            dict_angle_img_result[max_val] = top_left
    scores = scores[:, scores[1, :].argsort()]
    # scores = scores[scores[1,:].argsort()]
    # print(scores[0, :])
    #画出分数曲线
    #plt.scatter(scores[1, :], scores[0, :])
    # plt.show()

    max_scores = np.max(scores[0, :])
    #获取最高分数模板名称
    high_img_name = dict_angle_img_name[max_scores]
    #获取在一级模板上的logo位置
    four_point_old = high_img_name.split('_')[4:-1]

    #计算匹配出的新的模板位置
    old_point = []
    top_left = dict_angle_img_result[max_scores]
    for i in range(4):
        old_point.append(
            [int(four_point_old[i * 2]) - 708 + top_left[0], int(four_point_old[i * 2 + 1]) - 333 + top_left[1]])
    new_point = np.array(old_point)
    new_point = new_point.reshape((-1, 1, 2))
    print(new_point)
    cv.polylines(img, np.int32([new_point]), True, (0, 255, 255), 3)
    logo_result_file = result_dir + img_path.split('/')[-1]
    cv.imwrite(logo_result_file, img)


template_dir = '/home/lqy/Data/phone/mi_6/data-add15_0/white/cut/'
img_dir = '/home/lqy/Data/phone/mi_6/7_9/white_2/'
result_path = '/home/lqy/Data/phone/mi_6/detection_result_15_white2/'
crop_path = '/home/lqy/Data/phone/mi_6/detection_result_15_white2/cut/'

mkdir(result_path)
img_paths, img_names = eachFile(img_dir)
for img_path in img_paths:
    get_highesr_scores(template_dir, img_path, result_path)
