import cv2
im = cv2.imread('demo.PNG')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0, 0, 255), 3)
for i in contours:
    print(cv2.contourArea(i))
    x, y, w, h = cv2.boundingRect(i)
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("img", im)
cv2.waitKey(0)