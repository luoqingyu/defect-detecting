# -*- coding: UTF-8 -*-
# 多进程
import pandas as pd
import cv2 as cv
import os
import numpy  as np
from multiprocessing import Pool
from multiprocessing import Manager
from blob import get_detection_result
# from imtransform import  *
import csv


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
            dict_angle_img_name[max_val] = tmp_img      # 分数与名称
            dict_angle_img_result[max_val] = top_left           # 分数与匹配点
            # dict_scores_img_name[max_val] = tmp_img.split('_')[3]

        # print(template_img.split('/')[-1].split('_')[-2])


def get_highesr_scores(template_dir, img_path, result_dir,data_csv,Template_interval,angle_range):
    mkdir(result_dir + '/cut')
    mkdir(result_dir + '/process')
    mkdir(result_dir + '/show_error')



    # 多进程获取初始模板中的最高分数
    p = Pool(80)
    first_template_path =data_csv.iloc[::Template_interval]['adr'].values
    first_template_num = len(first_template_path)
    print(first_template_num)


    img = cv.imread(img_path, 0)
    img1 = img.copy
    dict_angle_img_name_fir = Manager().dict()
    dict_angle_img_result_fir = Manager().dict()
    y = range(first_template_num)
    list_scores_fir = Manager().list(y)  # 主进程与子进程共享这个List
    list_names_fir= Manager().list(y)
    for num, template_img in enumerate(first_template_path):
        p.apply_async(get_once_matching_result, args=(num, dict_angle_img_name_fir, dict_angle_img_result_fir,
                                                      list_scores_fir, list_names_fir, template_img, img))
        #         # print(tmp_img)
    p.close()
    p.join()

    numpy_scores_fir = np.array(list_scores_fir, dtype='float64')
    numpy_names_fir = np.array(list_names_fir, dtype='str')
    print(list_scores_fir)

    first_max_scores = np.max(numpy_scores_fir)
    print( np.where(numpy_scores_fir.max()))
    max_scores_path_first = numpy_names_fir[numpy_scores_fir.where(numpy_scores_fir.max())]


    print(first_max_scores)
    print(max_scores_path_first)

    first_math_angle = data_csv[data_csv['adr']==max_scores_path_first[0]]['angle'].values
    print('first_match_angle', first_math_angle)
    # 读取csv文件
    # match_points = data_csv.loc[abs(data_csv['angle'] - float(match_angle)) < 0.01]
    range_match = data_csv.loc[abs(data_csv['angle'] - float(first_math_angle)) < 0.5]
    adrs = range_match['adr'].values
    num_match = len(adrs)
    print('num_match',num_match)
    # print(num_match)
    # for i in range(adrs.shape[0]):
    #     #  print(adrs[i])
    #     # print(type(adrs))
    #     img_x = cv.imread(adrs[i], 0)
    #     # print(img_x)






    # 多进程深度匹配图像
    p = Pool(80)
    # dict_scores_img_name = Manager().dict()  # 主进程与子进程共享这个字典
    dict_angle_img_name = Manager().dict()  # 主进程与子进程共享这个字典
    dict_angle_img_result = Manager().dict()  # 主进程与子进程共享这个字典
    x = range(num_match)
    list_scores = Manager().list(x)  # 主进程与子进程共享这个List
    list_names = Manager().list(x)
    for num, tmp_img in enumerate(adrs):
        # print(tmp_img)
        p.apply_async(get_once_matching_result, args=(num, dict_angle_img_name, dict_angle_img_result,
                                                      list_scores, list_names, tmp_img, img))
        # print(tmp_img)
    p.close()
    p.join()
    numpy_scores = np.array(list_scores, dtype='float64')
    numpy_names = np.array(list_names, dtype='str')
    max_scores = np.max(numpy_scores)
    print('max_scores:',max_scores)
    max_scores_path = numpy_names[np.where(np.max(numpy_scores))]
    print('max_scores_path:',max_scores_path)
    # scores = np.concatenate((numpy_scores, numpy_names), axis=0)
    # scores = scores.reshape((2, num_match))
    # print(scores)
    # print(scores.dtype)
    # print(scores.shape)
    # scores = scores[:, scores[1, :].argsort()]
    # print(scores)
    # scores = scores[scores[1,:].argsort()]
    # print(scores[0, :])
    # python要用show展现出来图
    # plt.scatter(scores[1,     :], scores[0, :])
    # plt.show()
    # max_scores = np.max(scores[0, :])
    high_img_name = dict_angle_img_name[max_scores]
    match_points = data_csv.loc[data_csv['adr']== high_img_name ]
    # 获取在一级模板上的logo位置
    z = match_points.iloc[0, 1:-2].values
    top_left = dict_angle_img_result[max_scores]
    for point_num in range(int(z.shape[0] / 8)):
        point = z[point_num * 8:point_num * 8 + 8]
    # 计算匹配出的新的模板位置
        old_point = []
        for i in range(4):
            old_point.append([int(point[i * 2]) - 708 + top_left[0], int(point[i * 2 + 1]) - 333 + top_left[1]])
        logo_result_file = result_dir + img_path.split('/')[-1]
        # patch = extract_poly_patch(img, old_point)
        new_point = np.array(old_point)
        cv.rectangle(img, (np.min(new_point[:, 0]), np.min(new_point[:, 1])),
                     (np.max(new_point[:, 0]), np.max(new_point[:, 1])), (0, 255, 255), 1)
        new_point = new_point.reshape((-1, 1, 2))
        cv.polylines(img, np.int32([new_point]), True, (0, 255, 255), 1)
    logo_result_file = result_dir + img_path.split('/')[-1]
    cv.imwrite(logo_result_file, img)

    # 计算匹配出的图片和标准图片的差异
    high_score_img = cv.imread(high_img_name, 0)
    w_img, h_img = high_score_img.shape[::-1]
    match_img = img[top_left[1]:top_left[1] + h_img, top_left[0]:top_left[0] + w_img]
    result_sub = cv.absdiff(match_img, high_score_img)
    result_sub_path = result_dir + 'cut/' + img_path.split('/')[-1]
    cv.imwrite(result_sub_path, result_sub)

    # 对差值进行二值化
    ret, thresh2 = cv.threshold(result_sub, 95, 255, cv.THRESH_BINARY)
    process_result_sub_path = result_dir + 'process/' + img_path.split('/')[-1]
    cv.imwrite(process_result_sub_path, thresh2)

    # 显示检测到的缺陷
    thresh2 = get_detection_result(thresh2)
    process_result_sub_path = result_dir + 'show_error/' + img_path.split('/')[-1]
    cv.imwrite(process_result_sub_path, thresh2)


template_dir = '/home/wzx/data/phone/mi_6/data-add15_0/white/cut/'
img_dir = '/home/lqy/Data/phone/mi_6/7_9/white_2/'
result_path = '/home/wzx/data/phone/mi_6/img_detection_result_5000_white/'
data_csv = pd.read_csv("/home/wzx/many_angle_matching/csvData.csv")
data_csv.sort_values(by='angle')
# data_csv['angle'] = data_csv['angle'].astype(str)
mkdir(result_path)
img_paths, img_names = eachFile(img_dir)
for img_path in img_paths:

    print(img_path)
    get_highesr_scores(template_dir, img_path, result_path, data_csv,Template_interval = 168,angle_range = 0.5 )
