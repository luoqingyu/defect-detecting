import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

img_path = '/home/lqy/628/4/'
template = cv.imread('/home/lqy/628/4_cut.bmp', 0)
result_path = '/home/lqy/628/4_result/'
crop_path = '/home/lqy/628/4_result/cut/'
im_names = os.listdir(img_path)
w, h = template.shape[::-1]

# methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
#               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
methods = ['cv.TM_CCOEFF_NORMED']
#'cv.TM_CCORR_NORMED']  # 'cv.TM_CCOEFF', CCORR,SQDIFF not very good

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
        #print(res)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
            print(min_val)
        else:
            top_left = max_loc
            print(max_val)
        logo_result_file = crop_path + imname + 'logo' + '_' + meth + '.jpg'
        cv.imwrite(logo_result_file, img[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w])

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)

        img_result_file = result_path + str(max_val)+imname + '_' + '.jpg'
        cv.imwrite(img_result_file, img)
        print(img_result_file)

        '''plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()
        '''


