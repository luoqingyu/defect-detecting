# -*- coding: UTF-8 -*-
'''
根据一张图像模板，旋转该图片生成多角度模板
一级模板，涵盖logo的矩形框
二级模板，实际的logo框
'''
import cv2
import numpy  as np
import  random
import cv2 as cv
import os
import  math
import csv


def mkdir(path):  # 判断是否存在指定文件夹，不存在则创建
    # 引入模块
    import os
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
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


'''
#根据选择的角度和尺度创建图片
im_name:待处理的图片
angle: 图片旋转的角度
Scale：图片变换的尺度  这里默认为0，5代表尺度变化正负5%
num:生成图片的数目
result_dir:生成图片的位置
'''
def creat_img(im_name,angle, Scale,result_dir,writer,num=80,):
    #创建截取模板的文件夹
    cut_dir = result_dir +'cut'+'/'
    mkdir(cut_dir)
    #读取模板图片
    img = cv2.imread(im_name, 0)
    w_img, h_img = img.shape[::-1]
    #创建角度等差数列
    angles = np.linspace(-angle,angle,num)
    for i in angles:
        img1 = img.copy()
        #随机一个尺度变化
        change_Scale = random.uniform(1-Scale/100.,1+Scale/100.)
        # Scale_img = cv2.resize(img2,change_Scale)
        #旋转图片
        M = cv2.getRotationMatrix2D((w_img / 2, h_img / 2),i, change_Scale)
        dst = cv2.warpAffine(img1, M, (w_img, h_img))
        dst1 =dst.copy()
        #生成的多角度模板的名称
        img_result_file = result_dir  + str(i)+ '_' + str(change_Scale)+ '.bmp'
        #-10*10到选的框应该是cv2.rectangle(dst1, (708,333), (1669,717), (0, 0, 255), 3)
        #大框位置
        cv2.rectangle(dst1, (617,307), (1771,727), (0, 0, 255), 3)
        #实际模板的多边形矩阵top_left开始，顺时针
        #Logo位置
        Pts = np.array([[799,463],[1581,460],[1580,623],[798,620]], np.int32)
        #小字位置
        Pts_si = np.array([[984,463],  [1024,463],   [1024,504],   [984,504] ], np.int32)
        Pts_xiao = np.array([[1349,467],   [1390,467]  ,  [1390,508],   [1349,508]], np.int32)
        Pts_china = np.array([[801,516] ,   [913,516]   , [913,560 ]  ,    [801,560]], np.int32)
        Pts_de = np.array([[1520,522]  ,   [1577,522] ,  [1577,566],    [1520,566]], np.int32)
        Pts_16 = np.array([[842,573],      [ 886,573 ] ,  [886,611]   ,  [842,611]], np.int32)
        Pts_dt = np.array([[1474,578],      [1531,578],    [1531,620]  ,   [1474,620]], np.int32)
        #Pts = np.array([[610, 411], [2150,412], [2150, 561], [609,561]], np.int32)

        #计算实际模板对应的新的位置



        new_pts = []
        new_pts.append(str(i))
        str_four_point_pts =  get_many_romate_point(Pts,w_img , h_img ,i)
        str_four_point_si = get_many_romate_point(Pts_si, w_img, h_img, i)
        str_four_point_xiao = get_many_romate_point(Pts_xiao, w_img, h_img, i)
        str_four_point_china = get_many_romate_point(Pts_china, w_img, h_img, i)
        str_four_point_de = get_many_romate_point(Pts_de, w_img, h_img, i)
        str_four_point_dt = get_many_romate_point(Pts_dt, w_img, h_img, i)
        str_four_point_16 = get_many_romate_point(Pts_16, w_img, h_img, i)
        new_pts= new_pts+str_four_point_pts+str_four_point_si +str_four_point_xiao +str_four_point_china +str_four_point_de+str_four_point_16 +str_four_point_dt

        #保存一级模板
        cut_img_name = cut_dir + str(i) + '.bmp'
        new_pts.append(cut_img_name)
        writer.writerow(new_pts)
        cv.imwrite(cut_img_name, dst[333:717, 708:1669])

        #显示旋转后的二级模板
        # new_pts = np.array(new_pts)
        # new_pts1 = new_pts.reshape((-1, 1, 2))
        # cv2.polylines(dst1, np.int32([new_pts1]), True, (0, 255, 255), 3)  #画多边形框
        # cv.imwrite(img_result_file, dst1)



#计算一个点旋转后的映射位置
def get_romate_point(point,center,angle):
    new_point = []
    distence = []
    distence.append(point[0] - center[0])
    distence.append(point[1] - center[1])
    new_point.append( round(distence[0]* math.cos(angle) + distence[1] * math.sin(angle) + center[0]))
    new_point.append( round(-distence[0] * math.sin(angle) + distence[1]* math.cos(angle) + center[1]))
    return new_point

def get_many_romate_point(Pts,w_img , h_img,i):
    four_point_str=''
    new_pts = []
    #four_point_str = '_' + str(i) + '_'
    for row in Pts:
        point = get_romate_point(row, (w_img / 2, h_img / 2), i * math.pi / 180)
        new_pts.append(point[0])
        new_pts.append(point[1])
    return  new_pts


if __name__ == '__main__':
    #创建存储模板信息的csv文件
    csvFile = open("csvData.csv", "w")
    writer = csv.writer(csvFile)
    #需要存储多少个框的信息
    points = ["point1", "point2","point3","point4", "point5","point6","point7"]
    #csv文件每一列的名称
    row_name = ['angle']
    for point in points:
        for point_num in range(8):
            row_name.append(point+str(point_num))
    row_name.append('adr')
    #写入列名称
    writer.writerow(row_name)
    #存放模板文件夹位置
    first_dir = '/home/lqy/Data/phone/mi_6/chose/'
    second_full_dir,second_child_dir = eachFile(first_dir)
    angle = 15                #(-15,+15)，模板角度变化范围
    Scale = 0                 #(-Scale/10,Scale/10)   10代表尺度正负0.1
    num_img =  80             #creat how many image    生成多少模板
    #创建模板存放位置
    result_dir = '/home/lqy/Data/phone/mi_6/data-add-xiaozi'+str(angle)+'_'+str(Scale)+'/'
    mkdir(result_dir)


    for dir in second_full_dir:
        third_full_dir,third_child_dir = eachFile(dir+'/')
        x_dir = result_dir+dir.split('/')[-1]
        mkdir(x_dir)
        for  im_name in third_full_dir:
            creat_img(im_name,angle,Scale,x_dir+'/',writer,num_img )
    csvFile.close()



























