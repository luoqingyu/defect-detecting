import cv2
def change_num(x):
    img = cv2.imread('/home/lqy/python_scripy/defect-detecting/canny/data/demo.bmp', 0)  # 直接读为灰度图像
    # 二值化图像
    #ret, thresh1 = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    low = cv2.getTrackbarPos('Low', 'image')
    high = cv2.getTrackbarPos('High', 'image')
    ret, thresh2 = cv2.threshold(img, low, high,cv2.THRESH_BINARY )
    win = cv2.namedWindow('image', flags=0)
    cv2.imshow('image', thresh2)

# Create a black image, a window and bind the function to window
img = cv2.imread('/home/lqy/python_scripy/defect-detecting/canny/data/demo.bmp', 0)  # 直接读为灰度图像
cv2.namedWindow('image')
ret, thresh2 = cv2.threshold(img, 110, 250,cv2.THRESH_BINARY )
cv2.imshow('two mode', thresh2)
cv2.createTrackbar('Low','image',0,255,change_num)
cv2.createTrackbar('High','image',0,255,change_num)
if cv2.waitKey(0) == 27:     #ese  drop out
    cv2.destroyAllWindows()
cv2.destroyAllWindows()