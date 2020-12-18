import cv2
import os
import numpy as np

perspective_correction = None
perspective_correction_inv = None
perspective_trapezoid = None
warp_size = None
orig_size = None


def compute_perspective(width,height,pt1,pt2,pt3,pt4):
    
    global perspective_trapezoid, perspective_dest

    perspective_trapezoid = [(pt1[0], pt1[1]), (pt2[0], pt2[1]), (pt3[0], pt3[1]), (pt4[0], pt4[1])]
    src = np.float32([pt1, pt2, pt3, pt4])
    # widest side on the trapezoid
    x1 = pt1[0]
    x2 = pt4[0]
    # height of the trapezoid
    y1 = pt1[1]
    y2 = pt2[1]
    h = y1 - y2
    # The destination is a rectangle with the height of the trapezoid and the width of the widest side
    dst = np.float32([[x1, h], [x1, 0], [x2, 0], [x2, h]])
    perspective_dest = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

    global perspective_correction, perspective_correction_inv
    global warp_size, orig_size
    
    perspective_correction = cv2.getPerspectiveTransform(src,dst)
    perspective_correction_inv = cv2.getPerspectiveTransform(dst,src)
    
    warp_size = (width, h)
    orig_size = (width, height)


def warp(img, filename):
    img_persp = img.copy()

    cv2.line(img_persp, perspective_dest[0], perspective_dest[1], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[1], perspective_dest[2], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[2], perspective_dest[3], (255, 255, 255), 3)
    cv2.line(img_persp, perspective_dest[3], perspective_dest[0], (255, 255, 255), 3)

    cv2.line(img_persp, perspective_trapezoid[0], perspective_trapezoid[1], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[1], perspective_trapezoid[2], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[2], perspective_trapezoid[3], (0, 192, 0), 3)
    cv2.line(img_persp, perspective_trapezoid[3], perspective_trapezoid[0], (0, 192, 0), 3)

    return img_persp, cv2.warpPerspective(img, perspective_correction, warp_size, flags=cv2.INTER_LANCZOS4)

compute_perspective(1024, 600, [160, 425], [484, 310], [546, 310], [877, 425])


filename = 'img.png'
img_bgr = cv2.imread(filename)
cv2.imshow('img',cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR))

img_persp, img_warped = warp(img_bgr, filename)
cv2.imshow('img1',img_persp)


cv2.waitKey(0)
cv2.destroyAllWindows()