# -*- coding: UTF-8 -*-
import cv2
import numpy as np
img = np.zeros((512, 512, 3), dtype=np.uint8)
# 建立一张空白的图像
cv2.line(img, (10, 10), (510, 510), (0, 255, 0), 5)
# img:图像，起点坐标，终点坐标，颜色，线的宽度
cv2.circle(img, (50, 50), 10, (0, 0, 255), -1)
# img:图像，圆心坐标，圆半径，颜色，线宽度(-1：表示对封闭图像进行内部填满)
cv2.rectangle(img, (70, 80), (90, 100), (255, 0, 0), -1)
# img:图像,起点坐标,终点坐标,颜色,线宽度
cv2.ellipse(img, (150, 150), (10, 5), 0, 0, 180, (0, 127, 0), -1)
# img:图像,中心坐标，长短轴长度(长轴长度,短轴长度),旋转角度,显示的部分(0:起始角度,180:终点角度),颜色，线宽度
Pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
Pts = Pts.reshape((-1, 1, 2))
cv2.polylines(img, [Pts], True, (0, 255, 255), 33)
cv2.imwrite('result.jpg', img)