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


img_path= '/home/lqy/Data/phone/628-chhose/2/Image__2018-06-28__11-34-40.bmp'
#img_path = '/home/lqy/Data/phone/628-chhose/2/Image__2018-06-28__11-34-40.bmp'
img = cv.imread(img_path, 0)
template_dir = '/home/lqy/Data/phone/628-data-add10_5/si_cut/'
result_path = '/home/lqy/Data/phone/628-data-add10_5/si_cut_result/'
crop_path = '/home/lqy/Data/phone/628-data-add10_5/si_cut_result/cut/'
scores = np.zeros( (2,40) )
full_dir,dir = eachFile(template_dir)
for num,template_img in enumerate(full_dir):
    template = cv.imread(template_img, 0)
    w, h = template.shape[::-1]
    methods = ['cv.TM_CCOEFF_NORMED']
    for meth in methods:
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img, template, method)
        #print(res)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            #print(top_left)
            print('scores')
            print(max_val)
            scores[0][num] = max_val
        print('name'+template_img.split('/')[-1].split('_')[-2])
        scores[1][num] = template_img.split('/')[-1].split('_')[-2]
        print(top_left)
        logo_result_file = crop_path + template_img.split('/')[-1]
        cv.imwrite(logo_result_file, img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w])
        #
        bottom_right = (top_left[0] + w, top_left[1] + h)
        img1=img.copy()
        cv.rectangle(img1, top_left, bottom_right, (0, 0, 255), 3)
        #
        img_result_file = result_path + template_img.split('/')[-1]
        cv.imwrite(img_result_file, img1)
        #










scores = scores[:,scores[1,:].argsort()]

#scores = scores[scores[1,:].argsort()]
print(scores[0,:])
plt.scatter(scores[1,:],scores[0,:])
# python要用show展现出来图
plt.show()

