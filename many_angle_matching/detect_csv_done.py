# -*- coding: UTF-8 -*-
'''
对图片进行多角度模板匹配
'''
import pandas as pd
import cv2 as cv
import os
import numpy  as np
from multiprocessing import Pool
from multiprocessing import Manager
# from blob import get_detection_result
# from imtransform import  *
import time


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
num：这是第几个进程，第几个待测模板
dict_angle_img_name：角度和名称的对应字典
dict_angle_img_result：角度和撇皮结果的对应字典
list_scores：得分List
list_names:模板名称List
tmp_img：本次检测的模板名称
img：待检测图片
'''


def get_once_matching_result(num, dict_angle_img_name, dict_angle_img_result, list_scores,
                             list_names, tmp_img, img):
    template = cv.imread(tmp_img, 0)
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
            list_scores[num] = max_val
            list_names[num] = tmp_img
            dict_angle_img_name[max_val] = tmp_img  # 分数与名称
            dict_angle_img_result[max_val] = top_left  # 分数与匹配点
            # dict_scores_img_name[max_val] = tmp_img.split('_')[3]

        # print(template_img.split('/')[-1].split('_')[-2])


'''
img_path：待检测图片路径
result_dir：保存检测结果的路径
data_csv  ：存储模板信息的csv文件
Template_interval：第一次匹配的模板间隔
angle_range：第二次匹配的模板角度变化范围
'''


def get_highesr_scores(img_path, result_dir, data_csv, Template_interval, angle_range):
    # 计算时间
    starttime = time.time()
    # 多进程获取初始模板中的最高分数
    p = Pool(80)
    first_template_path = data_csv.iloc[::Template_interval]['adr'].values
    first_template_num = len(first_template_path)
    print(first_template_num)
    img_color = cv.imread(img_path)
    img = cv.imread(img_path, 0)
    img1 = img.copy()
    # 多进程需要使用的变量
    dict_angle_img_name_fir = Manager().dict()
    dict_angle_img_result_fir = Manager().dict()
    list_first_template_num = range(first_template_num)
    list_scores_fir = Manager().list(list_first_template_num)  # 主进程与子进程共享这个List
    list_names_fir = Manager().list(list_first_template_num)
    # 获取每张模板的匹配分数
    for num, template_img in enumerate(first_template_path):
        p.apply_async(get_once_matching_result, args=(num, dict_angle_img_name_fir, dict_angle_img_result_fir,
                                                      list_scores_fir, list_names_fir, template_img, img))
        # print(tmp_img)
    p.close()
    p.join()

    endtime = time.time()
    print('first match time', endtime - starttime)

    # 解析匹配的结果
    max_scores = max(list_scores_fir)
    print(max_scores)
    max_scores_path_first = list_names_fir[list_scores_fir.index(max(list_scores_fir))]
    # print(max_scores_path_first)
    first_math_angle = data_csv[data_csv['adr'] == max_scores_path_first]['angle'].values
    print('first_match_angle', first_math_angle)

    # 进行二次匹配
    # 读取csv文件
    # match_points = data_csv.loc[abs(data_csv['angle'] - float(match_angle)) < 0.01]
    range_match = data_csv.loc[abs(data_csv['angle'] - float(first_math_angle)) < angle_range]
    adrs = range_match['adr'].values
    num_match = len(adrs)
    print('num_match', num_match)

    # 多进程深度匹配图像
    p = Pool(80)
    dict_angle_img_name_second = Manager().dict()  # 主进程与子进程共享这个字典
    dict_angle_img_result_second = Manager().dict()  # 主进程与子进程共享这个字典
    x = range(num_match)
    list_scores_second = Manager().list(x)  # 主进程与子进程共享这个List
    list_names_second = Manager().list(x)
    for num, tmp_img in enumerate(adrs):
        p.apply_async(get_once_matching_result, args=(num, dict_angle_img_name_second, dict_angle_img_result_second,
                                                      list_scores_second, list_names_second, tmp_img, img))
    p.close()
    p.join()

    endtime = time.time()
    print('second match time',endtime - starttime)

    # 解析二次匹配结果
    max_scores_second = max(list_scores_second)
    print('max_scores_second', max_scores_second)
    max_scores_path_second = list_names_second[list_scores_second.index(max_scores_second)]
    high_img_name = dict_angle_img_name_second[max_scores_second]
    print('max_scores_path_second' + max_scores_path_second)
    match_points = data_csv.loc[data_csv['adr'] == high_img_name]
    # 获取在一级模板上的logo位置
    z = match_points.iloc[0, 1:-2].values
    print(z)
    top_left = dict_angle_img_result_second[max_scores_second]
    for point_num in range(int(z.shape[0] / 8)):
        point = z[point_num * 8:point_num * 8 + 8]
        # 计算匹配出的新的模板位置
        old_point = []
        for i in range(4):
            old_point.append([int(point[i * 2]) - 708 + top_left[0], int(point[i * 2 + 1]) - 333 + top_left[1]])
        new_point = np.array(old_point)
        cv.rectangle(img, (np.min(new_point[:, 0]), np.min(new_point[:, 1])),
                     (np.max(new_point[:, 0]), np.max(new_point[:, 1])), (0, 255, 255), 1)
        new_point = new_point.reshape((-1, 1, 2))
        cv.polylines(img, np.int32([new_point]), True, (0, 255, 255), 1)
    logo_result_file = result_dir + img_path.split('/')[-1]
    cv.imwrite(logo_result_file, img)

    # 对检测结果进行处理
    # 计算匹配出的图片和标准图片的差异
    high_score_img = cv.imread(high_img_name, 0)
    w_img, h_img = high_score_img.shape[::-1]
    match_img = img1[top_left[1]:top_left[1] + h_img, top_left[0]:top_left[0] + w_img]
    match_img_color = img_color[top_left[1]:top_left[1] + h_img, top_left[0]:top_left[0] + w_img, :]
    result_sub = cv.absdiff(match_img, high_score_img)
    result_sub_path = result_dir + 'cut/' + img_path.split('/')[-1]
    cv.imwrite(result_sub_path, result_sub)

    # 对差值进行二值化
    ret, thresh2 = cv.threshold(result_sub, 95, 255, cv.THRESH_BINARY)
    process_result_sub_path = result_dir + 'process/' + img_path.split('/')[-1]
    cv.imwrite(process_result_sub_path, thresh2)

    # 去除顶帽
    ret, high_score_img = cv.threshold(high_score_img, 60, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    # 腐蚀图像
    eroded = cv.erode(high_score_img, kernel)
    # 膨胀图像
    kerne2 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilated = cv.dilate(high_score_img, kerne2)
    chazhi = dilated - eroded
    dingmao_path = result_dir + 'dingmao/' + img_path.split('/')[-1]
    cv.imwrite(dingmao_path, chazhi)

    # 处理差值
    chazhi[chazhi > 0] = 1
    chazhi = 1 - chazhi
    dingmao_reult = result_sub * chazhi
    dingmao_reult = cv.blur(dingmao_reult, (3, 3))
    ret, thresh3 = cv.threshold(dingmao_reult, 50, 255, cv.THRESH_BINARY)
    dingmao_reult_path = result_dir + 'dingmao_result/' + img_path.split('/')[-1]
    cv.imwrite(dingmao_reult_path, dingmao_reult)
    dingmao_erzhihua_path = result_dir + 'dingmao_erzhihua/' + img_path.split('/')[-1]
    cv.imwrite(dingmao_erzhihua_path, thresh3)

    result_add = thresh3 + thresh2
    result_add_path_path = result_dir + 'add_result/' + img_path.split('/')[-1]
    cv.imwrite(result_add_path_path, result_add)

    # 显示检测到的缺陷
    show_result_path = result_dir + 'show_error/' + img_path.split('/')[-1]
    # result_add = cv.cvtColor(result_add, cv.COLOR_BGR2GRAY)
    # ret, thresh4 = cv.threshold(result_add, 10, 255, 0)
    im2, contours, hierarchy = cv.findContours(result_add, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(match_img_color, contours, -1, (0, 0, 255), 1)
    cv.imwrite(show_result_path, match_img_color)
    # 计算检测用时
    endtime = time.time()
    print(endtime - starttime)


def main():
    img_dir = '/home/lqy/Data/phone/mi_6/7_9/white_2/'
    result_path = '/home/lqy/Data/phone/mi_6/re_5000_60_0.3/'
    data_csv = pd.read_csv("./csvData.csv")

    # 创建存放结果的文件夹
    mkdir(result_path)
    mkdir(result_path + 'cut')
    mkdir(result_path + 'process')
    mkdir(result_path + 'show_error')
    mkdir(result_path + 'dingmao')
    mkdir(result_path + 'dingmao_result')
    mkdir(result_path + 'dingmao_erzhihua')
    mkdir(result_path + 'add_result')

    data_csv.sort_values(by='angle')
    # data_csv['angle'] = data_csv['angle'].astype(str)
    # 获取所有需要检测的图片名称
    img_paths, img_names = eachFile(img_dir)
    for img_path in img_paths:
        print(img_path)
        get_highesr_scores(img_path, result_path, data_csv, Template_interval=60, angle_range=0.3)


if __name__ == "__main__":
    main()
