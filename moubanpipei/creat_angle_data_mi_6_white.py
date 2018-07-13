# -*- coding: UTF-8 -*-
import cv2
import numpy  as np
import  random
# 读取图像
import cv2 as cv
import os
import  math
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
def creat_img(im_name,angle, Scale,result_dir):
    cut_dir = result_dir +'cut'+'/'
    mkdir(cut_dir)
    img = cv2.imread(im_name, 0)
    w_img, h_img = img.shape[::-1]
    angles = np.linspace(-angle,angle,80)
    for i in angles:
        img1 = img.copy()
        change_Scale = random.uniform(1-Scale/100.,1+Scale/100.)
        # Scale_img = cv2.resize(img2,change_Scale)
        M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2),i, change_Scale)
        dst = cv2.warpAffine(img1, M, (w_img, h_img))
        dst1 =dst.copy()
        result_dir =  result_dir
        img_result_file = result_dir  + str(i)+ '_' + str(change_Scale)+ '.bmp'
        #-10*10到
        #cv2.rectangle(dst1, (708,333), (1669,717), (0, 0, 255), 3)
        #-20-20
        cv2.rectangle(dst1, (617,307), (1771,727), (0, 0, 255), 3)

        Pts = np.array([[799,463],[1581,460],[1580,623],[798,620]], np.int32)
        #647，301    1723   725


        cut_img_name = cut_dir + str(i)
        #Pts = np.array([[610, 411], [2150,412], [2150, 561], [609,561]], np.int32)

        new_pts = []
        four_point_str ='_'+str(i)+'_'
        for row in Pts:
            point=get_romate_point(row, (w_img / 2, h_img / 2), i * math.pi / 180)
            new_pts.append(point)
            four_point_str = four_point_str + str(int(point[0]))+'_'+str(int(point[1]))+'_'
        new_pts = np.array(new_pts)
        cut_img_name = cut_dir + four_point_str+ '.bmp'
        cv.imwrite(cut_img_name, dst[333:717,708:1669])
        new_pts1 = new_pts.reshape((-1, 1, 2))
        cv2.polylines(dst1, np.int32([new_pts1]), True, (0, 255, 255), 3)
        cv.imwrite(img_result_file, dst1)

def get_result(img_path,template_path,result_path,crop_path):
    template = cv.imread(template_path, 0)
    im_names = os.listdir(img_path)
    w, h = template.shape[::-1]
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    methods = ['cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR_NORMED']  # 'cv.TM_CCOEFF', CCORR,SQDIFF not very good
    for im_name in im_names:
        imname = os.path.splitext(im_name)[0]
        ext = os.path.splitext(im_name)[1]
        # if ext not in ['.jpg', '.JPG', '.bmp']:
        if ext not in ['.bmp']:
            continue
        im_file = os.path.join(img_path, im_name)
        img = cv.imread(im_file, 0)
        w_img, h_img = img.shape[::-1]
        img2 = img.copy()
        # All the 6 methods for comparison in a list
        for meth in methods:
            img = img2.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            logo_result_file = crop_path + imname + 'logo' + '_' + meth + '.bmp'
            cv.imwrite(logo_result_file, img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w])

            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)

            img_result_file = result_path + imname + '_' + meth + '.bmp'
            cv.imwrite(img_result_file, img)
            print(img_result_file)

def get_romate_point(point,center,angle):
    new_point = []
    distence = []
    distence.append(point[0] - center[0])
    distence.append(point[1] - center[1])
    new_point.append( round(distence[0]* math.cos(angle) + distence[1] * math.sin(angle) + center[0]))
    new_point.append( round(-distence[0] * math.sin(angle) + distence[1]* math.cos(angle) + center[1]))
    return new_point


if __name__ == '__main__':
    cut_dir = '/home/lqy/Data/phone/628_cut/'
    first_dir = '/home/lqy/Data/phone/mi_6/chose/'
    second_full_dir,second_child_dir = eachFile(first_dir)
    angle = 15                   #(-5,+5)
    Scale = 0                 #(-Scale/10,Scale/10)
    num_img =  100              #creat how many image
    result_dir = '/home/lqy/Data/phone/mi_6/data-add'+str(angle)+'_'+str(Scale)+'/'
    mkdir(result_dir)
    detect_result_dir  =  result_dir.replace('data','result_data')
    #mkdir(detect_result_dir)
    for dir in second_full_dir:
        third_full_dir,third_child_dir = eachFile(dir+'/')
        x_dir = result_dir+dir.split('/')[-1]
        mkdir(x_dir)
        for  im_name in third_full_dir:
            creat_img(im_name,angle,Scale,x_dir+'/')
            #cut_img =cut_dir + im_name.split('/')[-2]+'/'+im_name.split('/')[-1]
            #every_detect_result_dir = detect_result_dir + im_name.split('/')[-2]+'/'
            #mkdir(every_detect_result_dir)
            #every_cut_detect_result_dir  = every_detect_result_dir + 'cut' +'/'
            #mkdir(every_cut_detect_result_dir)
            #get_result( result_dir+im_name.split('/')[-2]+'/', cut_img, every_detect_result_dir,every_cut_detect_result_dir)































