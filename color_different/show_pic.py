#-*- coding: UTF-8 -*-
import  cv2
import numpy as np
def get_sum_RGB(image):
    # 加载图像
    image = cv2.imread(image)

    # 通道分离，注意顺序BGR不是RGB
    (B, G, R) = cv2.split(image)
    cv2.imshow("orign", image)
# 显示各个分离出的通道
    cv2.imshow("Red", R)
    cv2.imshow("Green", G)
    cv2.imshow("Blue", B)
    return (np.sum(R),np.sum(G),np.sum(B))


if __name__ == '__main__':
    blue = '/home/lqy/python_scripy/defect-detecting/color_different/data/demo.jpg'
    #black = '/home/lqy/python_scripy/defect-detecting/color_different/data/black.jpg'
    #pink = '/home/lqy/python_scripy/defect-detecting/color_different/data/pink.jpg'
    #brown = '/home/lqy/python_scripy/defect-detecting/color_different/data/brown.jpg'
    print(get_sum_RGB(blue))
    #print(get_sum_RGB(black))
    #print(get_sum_RGB(pink))
    #print(get_sum_RGB(brown))
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
