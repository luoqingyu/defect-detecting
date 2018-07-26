# -*- coding: UTF-8 -*-
# !/usr/bin/python
# Standard imports
import cv2
import numpy as np
def get_detection_result(img):
    #im = cv2.imread("demo.bmp", cv2.IMREAD_GRAYSCALE)
    img = 255-img
# Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 0
    #二值化
    # Change thresholds
    params.minThreshold = 95
    params.maxThreshold = 200
    #最小面积
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0
    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(img)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show blobs

    return  im_with_keypoints

def main():
  img = cv2.imread("demo.bmp", cv2.IMREAD_GRAYSCALE)
  img = get_detection_result(img)
  cv2.imshow("Keypoints", img)
  cv2.waitKey(0)
if __name__ == '__main__':
  main()