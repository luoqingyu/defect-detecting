import cv2 as cv
import os
import matplotlib.pyplot as plt
import  numpy  as np
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
def get_highesr_scores(template_dir,img_path,result_dir):

    img = cv.imread(img_path, 0)
    scores = np.zeros((2, 80))
    full_dir, dir = eachFile(template_dir)
    dict_angle_img_name = {}
    dict_angle_img_result = {}
    for num, template_img in enumerate(full_dir):
        template = cv.imread(template_img, 0)
        w, h = template.shape[::-1]
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
                scores[0][num] = max_val

            #print(template_img.split('/')[-1].split('_')[-2])
            scores[1][num] = template_img.split('_')[3]
            dict_angle_img_name[max_val]= template_img
            dict_angle_img_result[max_val] = top_left
    scores = scores[:, scores[1, :].argsort()]
    print(scores)
    # scores = scores[scores[1,:].argsort()]
    #print(scores[0, :])
    plt.scatter(scores[1,     :], scores[0, :])

    max_scores = np.max(scores[0,:])
    print()
    # python要用show展现出来图
    #plt.show()
    print(dict_angle_img_name)
    print(scores)
    high_img_name = dict_angle_img_name[max_scores]
    four_point_old =  high_img_name.split('_')[4:-1]
    old_point =[]
    top_left =dict_angle_img_result[ max_scores]
    for i in range(4):
        old_point.append([int(four_point_old[i*2])-708+top_left[0],int(four_point_old[i*2+1])- 333 + top_left[1]])
    new_point = np.array(old_point)
    new_point = new_point.reshape((-1, 1, 2))
    print(new_point)
    cv.polylines(img, np.int32([new_point]), True, (0, 255, 255), 3)
    logo_result_file = result_dir + img_path.split('/')[-1]
    cv.imwrite(logo_result_file, img)


template_dir = '/home/lqy/Data/phone/mi_6/data-add15_0/white/cut/'
img_dir  ='/home/lqy/Data/phone/mi_6/7_9/white_2/'
result_path = '/home/lqy/Data/phone/mi_6/detection_result_15_white2/'
crop_path = '/home/lqy/Data/phone/mi_6/detection_result_15_white2/cut/'

mkdir(result_path)
img_paths,img_names = eachFile(img_dir)
for img_path in img_paths:
    get_highesr_scores(template_dir,img_path,result_path)





