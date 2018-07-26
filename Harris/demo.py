import cv2
import numpy as np
filename = './data/mi4.JPG'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(gray, 116, 250,cv2.THRESH_BINARY )
cv2.imshow('thresh2',thresh2)

#canny算子检测边缘
canny_img = cv2.Canny(thresh2, 100, 100)
canny_img1=canny_img
win1 = cv2.namedWindow('canny', flags=0)
cv2.imshow('canny', canny_img)






thresh2= np.float32(canny_img)
#图像转换为float32
dst = cv2.cornerHarris(thresh2 ,2 ,3 , 0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)#图像膨胀
# Threshold for an optimal value, it may vary depending on the image.
#print(dst)
#img[dst>0.00000001*dst.max()]=[0,0,255] #可以试试这个参数，角点被标记的多余了一些
img[dst>0.3*dst.max()]=[0,0,255]#角点位置用红色标记
#这里的打分值以大于0.01×dst中最大值为边界

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
