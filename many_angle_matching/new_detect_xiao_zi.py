# -*- coding: UTF-8 -*-
# 多进程
import pandas as pd
import cv2 as cv
import os
import numpy  as np
from multiprocessing import Pool
from multiprocessing import Manager
from  blob import get_detection_result
from imtransform import  *


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

def show_error(im):

    im2, contours, hierarchy = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(im, contours, -1, (0, 0, 255), 3)
    for i in contours:
        print(cv.contourArea(i))
        x, y, w, h = cv.boundingRect(i)
        cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return im
    #cv.imshow("img", im)
    #cv.waitKey(0)
def get_once_matching_result(num, dict_angle_img_name, dict_angle_img_result,dict_scores_img_name, list_scores, list_names, template_img,
                             img):
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
            list_scores[num] = max_val
            list_names[num] = template_img.split('_')[3]
            dict_angle_img_name[max_val] = template_img         #分数与名称
            dict_angle_img_result[max_val] = top_left           #分数与匹配点
            dict_scores_img_name[max_val]  =  template_img.split('_')[3]


        # print(template_img.split('/')[-1].split('_')[-2])


def get_highesr_scores(template_dir, img_path, result_dir,data_csv):
    p = Pool(80)
    mkdir( result_dir + '/cut')
    mkdir(result_dir + '/process')
    mkdir(result_dir + '/show_error')
    mkdir(result_dir + '/xiaozi')
    img = cv.imread(img_path, 0)
    img1 = img.copy
    full_dir, dir = eachFile(template_dir)
    dict_scores_img_name = Manager().dict()  # 主进程与子进程共享这个字典
    dict_angle_img_name = Manager().dict()  # 主进程与子进程共享这个字典
    dict_angle_img_result = Manager().dict()  # 主进程与子进程共享这个字典
    x = range(80)
    list_scores = Manager().list(x)  # 主进程与子进程共享这个List
    list_names = Manager().list(x)
    for num, template_img in enumerate(full_dir):
        p.apply_async(get_once_matching_result, args=(num, dict_angle_img_name, dict_angle_img_result,dict_scores_img_name,
                                                      list_scores, list_names, template_img, img))
    p.close()
    p.join()
    numpy_scores = np.array(list_scores, dtype='float64')
    numpy_names = np.array(list_names, dtype='float64')

    scores = np.concatenate((numpy_scores, numpy_names), axis=0)
    scores = scores.reshape((2, 80))
    # print(scores.dtype)
    # print(scores.shape)
    scores = scores[:, scores[1, :].argsort()]
    # print(scores)
    # scores = scores[scores[1,:].argsort()]
    # print(scores[0, :])
    # plt.scatter(scores[1,     :], scores[0, :])

    max_scores = np.max(scores[0, :])
    match_angle = dict_scores_img_name[max_scores]

    match_points = data_csv.loc[abs(data_csv['angle'] - float(match_angle)) < 0.01]
    #fanwei_match = data_csv.loc[abs(data_csv['angle'] - float(match_angle)) < 1]
    #adrs = fanwei_match['adr']

    # python要用show展现出来图
    # plt.show()
    top_left = dict_angle_img_result[max_scores]
    high_img_name = dict_angle_img_name[max_scores]

    # 解析所有匹配到的点
    z = match_points.iloc[0, 1:-2].values
    for point_num in range(int(z.shape[0] / 8)):
        point = z[point_num * 8:point_num * 8 + 8]
        # print(point)
        old_point = []
        for i in range(4):
            old_point.append([int(point[i * 2]) - 708 + top_left[0], int(point[i * 2 + 1]) - 333 + top_left[1]])
        logo_result_file = result_dir + img_path.split('/')[-1]
        #patch = extract_poly_patch(img, old_point)
        new_point = np.array(old_point)
        cv.rectangle(img, (np.min(new_point[:, 0]), np.min(new_point[:, 1])),
                     (np.max(new_point[:, 0]), np.max(new_point[:, 1])), (0, 255, 255), 1)

        new_point = new_point.reshape((-1, 1, 2))
        cv.polylines(img, np.int32([new_point]), True, (0, 255, 255), 1)

    logo_result_file = result_dir + img_path.split('/')[-1]
    cv.imwrite(logo_result_file, img)
    #result_sub_path =  result_dir +'cut/'+ img_path.split('/')[-1]
    #cv.imwrite(result_sub_path, img)
    #xiaozi_file = result_dir + 'show_error/' + str(point_num) + '.bmp'
    #cv.imwrite(xiaozi_file, patch)

    # 计算匹配出的图片和标准图片的差异
    high_score_img = cv.imread(high_img_name, 0)
    w_img, h_img = high_score_img.shape[::-1]
    match_img = img1[top_left[1]:top_left[1] + h_img, top_left[0]:top_left[0] + w_img]
    result_sub = cv.absdiff(match_img, high_score_img)


    result_sub_path = result_dir + 'cut/' + img_path.split('/')[-1]
    cv.imwrite(result_sub_path, result_sub)
    # 对差值进行二值化
    ret, thresh2 = cv.threshold(result_sub, 95, 255, cv.THRESH_BINARY)
    process_result_sub_path = result_dir + 'process/' + img_path.split('/')[-1]
    cv.imwrite(process_result_sub_path, thresh2)
    # 显示检测到的缺陷
    thresh2 = show_error(thresh2)
    #process_result_sub_path = result_dir + 'show_error/' + img_path.split('/')[-1]
    cv.imwrite(process_result_sub_path, thresh2)







template_dir = '/home/lqy/Data/phone/mi_6/data-add15_0/white/cut/'
img_dir = '/home/lqy/Data/phone/mi_6/7_9/white_2/'
result_path = '/home/lqy/Data/phone/mi_6/xiaozi_bad_img_detection_result_15_white2/'
data_csv = pd.read_csv("/home/lqy/python_scripy/defect-detecting/many_angle_matching/csvData.csv")
#data_csv['angle'] = data_csv['angle'].astype(str)
mkdir(result_path)
img_paths, img_names = eachFile(img_dir)
for img_path in img_paths:
    get_highesr_scores(template_dir, img_path, result_path,data_csv)
