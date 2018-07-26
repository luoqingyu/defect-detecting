# Copyright 2018 The WTI Alchemist. All Rights Reserved.
# Author: Shuo Chang
# Email: changshuo@bupt.edu.cn
import cv2
import numpy as np
import imutils
import math
import matplotlib.pyplot as plt


def ensure_points_inside_bounds(points, w, h):
    """
    If the coordinate of points exceeds the image region,
    then truncate it and make sure it inside the image region

    Args:
        points: list with storing 8 elements, and two of them from
        left to right (with no overlap) makes a point: x ,y
        w: the width of current image
        h: the height of current image

    Returns:
        points_: points truncated by this function
    """
    points_ = []
    for p_index in range(4):
        for xy in range(2):
            i = p_index * 2 + xy
            cor_val = points[i]
            if xy == 0:
                if cor_val < 1:
                    cor_val = 1
                elif cor_val > w:
                    cor_val = w
            else:
                if cor_val < 1:
                    cor_val = 1
                elif cor_val > h:
                    cor_val = h
            points_.append(cor_val)

    return points_


def convert_to_clockwise(points):
    """
    Covert the anti-clockwise points (4) to clockwise

    Args:
        points: the order of it is anti-clockwise

    Returns:
        res: convert the input points as clockwise and return as numpy array with the shape as (4, 2)

    Reference:
    [1]:https://gamedev.stackexchange.com/questions/13229/sorting-array-of-points-in-clockwise-order
    [2]:
    """
    poly = np.asarray(points).astype(dtype=np.float32)
    poly = np.reshape(poly, [4, 2])

    barycenter = np.mean(poly, axis=0)
    pb_dxdy = poly - barycenter
    points_angle = np.arctan2(pb_dxdy[:, 1], pb_dxdy[:, 0])
    sort_index = np.argsort(points_angle)

    res = poly[sort_index, :]

    return res.astype(dtype=np.int)


def ensure_point_in_region(im_shape, x, y, w, h):
    """
    Ensure the bounding box (x, y, w, h) inside the image region

    Args:
        im_shape: im_shape[0] is image height, im_shape[1] is image width
        x: the up left point x of rectangle
        y: the up left point y of rectangle
        w: the width of rectangle
        h: the height of rectangle

    Returns:
        x1: the up left point x index of rectangle
        y1: the up left point y index of rectangle
        x2: the bottom right point x index of rectangle
        y2: the bottom right point y index of rectangle
    """
    x1 = x
    x2 = x + w
    y1 = y
    y2 = y + h
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > im_shape[1]:
        x2 = im_shape[1]
    if y2 > im_shape[0]:
        y2 = im_shape[0]

    return x1, x2, y1, y2


def get_region(im_shape, w, h):
    """
    Get the rotated patch region inside the image region, the patch center is the same as the image center

    Args:
       im_shape: im_shape[0] is image height, im_shape[1] is image width
       w: the width of rectangle
       h: the height of rectangle

    Returns:
       x1: the up left point x index of rectangle
       y1: the up left point y index of rectangle
       x2: the bottom right point x index of rectangle
       y2: the bottom right point y index of rectangle
    """
    x1 = int(round((im_shape[1] - w) / 2.0, 0))
    y1 = int(round((im_shape[0] - h) / 2.0, 0))
    x2 = int(round(x1 + w - 1, 0))
    y2 = int(round(y1 + h - 1, 0))

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > im_shape[1]:
        x2 = im_shape[1]
    if y2 > im_shape[0]:
        y2 = im_shape[0]

    return x1, x2, y1, y2


def extract_poly_patch(im, points):
    """
    Extract poly patch from im region

    Args:
        im: im data return by cv2.imread()
        points: list with storing 8 elements, and two of them from
        left to right (with no overlap) makes a point: x ,y

    Returns:
        patch: rotate the patch region defines by points, after that extract it
    """
    #points = ensure_points_inside_bounds(points, im_w, im_h)
    poly_region = convert_to_clockwise(points)

    im_mask = np.copy(im)
    im_mask = im_mask.astype(dtype=np.float32)
    im_mask = im_mask + 255
    im_mask[im_mask > 255] = 255
    im_mask = im_mask.astype(np.uint8)
    im_save = np.copy(im)
    rect = cv2.minAreaRect(poly_region)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(box)

    # visualize the results
    cv2.drawContours(im, [box], 0, (0, 0, 255), 10)
    cv2.rectangle(im, (x-10, y-10), (x + w + 20, y + h + 20), (0, 255, 0), 10)
    cv2.imwrite('vis.jpg', im)

    im_mask = cv2.drawContours(im_mask, [box], 0, (0, 0, 0), -1)
    img2gray = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    im_fg = cv2.bitwise_and(im_save, im_save, mask=mask_inv)

    x1, x2, y1, y2 = ensure_point_in_region(im_fg.shape[:2], x, y, w, h)

    im_fg = im_fg[y1:y2, x1:x2, :]

    if not im_fg.size:
        return None
    else:
        im_rotated = imutils.rotate_bound(im_fg, -1*rect[2])

        x1, x2, y1, y2 = get_region(im_rotated.shape[:2], rect[1][0], rect[1][1])
        patch = im_rotated[y1:y2, x1:x2, :]

        if not patch.size:
            return None
        else:
            return patch


def vis_extract_poly_patch(im, center, line_len, points):
    """
    Debug the procedure of "Extract poly patch from im region"

    Args:
        im: im data return by cv2.imread()
        center: the poly center (x, y) coordinates
        line_len: the center distance to four corners
        points: list with storing 8 elements, and two of them from
        left to right (with no overlap) makes a point: x ,y

    Returns:
        im_vis: add straight rectangle and bounding rectangle which is drawn with minimum area
        im_rotated: rotate part of im region with the poly center as origin
        patch: rotate the patch region defines by points, after that extract it
    """
    assert (len(points) == 8)
    im_h = im.shape[0]
    im_w = im.shape[1]
    points = ensure_points_inside_bounds(points, im_w, im_h)
    poly_region = convert_to_clockwise(points)

    im_mask = np.copy(im)
    im_vis = np.copy(im)
    im_mask = im_mask.astype(dtype=np.float32)
    im_mask = im_mask + 255
    im_mask[im_mask > 255] = 255
    im_mask = im_mask.astype(np.uint8)
    im_save = np.copy(im)
    rect = cv2.minAreaRect(poly_region)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(box)

    # visualize the results
    cv2.drawContours(im_vis, [box], 0, (0, 0, 255), 30)
    cv2.rectangle(im_vis, (x-10, y-10), (x + w + 20, y + h + 20), (0, 255, 0), 30)
    cv2.circle(im_vis, center, 30, (255, 0, 0), -1)
    cv2.circle(im_vis, center, line_len, (255, 255, 255), 30)

    im_mask = cv2.drawContours(im_mask, [box], 0, (0, 0, 0), -1)
    img2gray = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)

    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    im_fg = cv2.bitwise_and(im_save, im_save, mask=mask_inv)

    x1, x2, y1, y2 = ensure_point_in_region(im_fg.shape[:2], x, y, w, h)

    im_fg = im_fg[y1:y2, x1:x2, :]

    im_rotated = imutils.rotate_bound(im_fg, -1*rect[2])

    x1, x2, y1, y2 = get_region(im_rotated.shape[:2], rect[1][0], rect[1][1])
    patch = im_rotated[y1:y2, x1:x2, :]
    im_vis = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
    im_rotated = cv2.cvtColor(im_rotated, cv2.COLOR_BGR2RGB)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    return im_vis, im_rotated, patch


def test():
    im_path = 'IMG_2565.JPG'
    im = cv2.imread(im_path)
    center = (2615, 1916)
    theta_range = 360
    theta = 20
    line_len = 1000
    fig = plt.figure(figsize=(1, 3))
    for i in range(theta_range):
        theta1 = math.radians(180 - theta/2 + i)
        theta2 = math.radians(180 + theta/2 + i)
        theta3 = math.radians(360 - theta/2 + i)
        theta4 = math.radians(360 + theta/2 + i)
        x1 = math.cos(theta1) * line_len + center[0]
        y1 = math.sin(theta1) * line_len + center[1]
        x2 = math.cos(theta2) * line_len + center[0]
        y2 = math.sin(theta2) * line_len + center[1]
        x3 = math.cos(theta3) * line_len + center[0]
        y3 = math.sin(theta3) * line_len + center[1]
        x4 = math.cos(theta4) * line_len + center[0]
        y4 = math.sin(theta4) * line_len + center[1]
        raw_im, rotated_im, patch = vis_extract_poly_patch(im, center, line_len, [x1, y1, x2, y2, x3, y3, x4, y4])
        fig.add_subplot(3, 1, 1)
        plt.imshow(raw_im)
        fig.add_subplot(3, 1, 2)
        plt.imshow(rotated_im)
        fig.add_subplot(3, 1, 3)
        plt.imshow(patch)
        plt.pause(0.001)
    plt.show()
    plt.close()


def main():
    im_path = 'IMG_2565.JPG'
    im = cv2.imread(im_path)
    points = [1770, 3716, 1770, 116, 3460, 116, 3460, 3716]
    patch = extract_poly_patch(im, points)
    cv2.imwrite('test.jpg', patch)


if __name__ == '__main__':
    test()
