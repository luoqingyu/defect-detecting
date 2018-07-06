# -*- coding: UTF-8 -*-
import  cv2
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


if __name__ == '__main__':
    img_dir = '/home/lqy/Data/phone/628-data-add10_5/4/'
    result_dir = '/home/lqy/Data/phone/628-data-add10_5/4_cut/'
    second_full_dir,second_child_dir = eachFile(img_dir)
    mkdir(result_dir)
    for dir in second_full_dir:
        img = cv2.imread(dir, 0)
        img_cropped = img[941:1171, 1062:1390]
        change_dir = result_dir + dir.split('/')[-1]
        cv2.imwrite(change_dir,img_cropped)











