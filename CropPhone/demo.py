#-*- coding: UTF-8 -*-
import  cv2
from get_point import  *
from imtransform import  *
import matplotlib.pyplot as plt
import numpy as np
def main():
    img_path = '/home/public/Datas/phone/hei/IMG_2565.JPG'
    im = cv2.imread(img_path)
    line1 = process_img(img_path = img_path )     #处理图片
    jiaodian_quchong = get_point(line1)      #获取交点
    print(jiaodian_quchong)
    # im = cv2.imread(img_path)
    points = []
    for point in jiaodian_quchong:
        points.append(int(point[0]*10))
        points.append(int(point[1]*10))

    a = 10*np.array([[jiaodian_quchong[0], jiaodian_quchong[1], jiaodian_quchong[2], jiaodian_quchong[3]]], dtype=np.int32)
    print(a)
    cv2.polylines(im,a, 1, 255,10)
    cv2.namedWindow("point", 0)
    cv2.resizeWindow("point", 640, 480)
    cv2.imshow('point', im)
    print(points)
    extract_poly_patch(im, points)
    if cv2.waitKey(0) == 27:     #ese  drop out
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

