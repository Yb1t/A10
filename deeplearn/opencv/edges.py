# import cv2
# img = cv2.imread("output.png")
# edges = cv2.Canny(img, 50, 150, apertureSize = 3)
# cv2.imshow("canny",edges)
# cv2.waitKey()

"""
cv2.HoughLines()
	dst:   输出图像. 它应该是个灰度图 (但事实上是个二值化图)
	lines: 储存着检测到的直线的参数对 (r,\theta) 的容器
	rho : 参数极径 r 以像素值为单位的分辨率. 我们使用 1 像素.
	theta: 参数极角 \theta 以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
	threshold:    设置阈值： 一条直线所需最少的的曲线交点
	srn and stn:  参数默认为0

cv2.HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 )
	dst:    输出图像. 它应该是个灰度图 (但事实上是个二值化图)
	lines:  储存着检测到的直线的参数对 (x_{start}, y_{start}, x_{end}, y_{end}) 的容器
	rho :   参数极径 r 以像素值为单位的分辨率. 我们使用 1 像素.
	theta:  参数极角 \theta 以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
	threshold:    设置阈值： 一条直线所需最少的的曲线交点。超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
	minLinLength: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.
	maxLineGap:   能被认为在一条直线上的两点的最大距离。
"""
import cv2
import numpy as np

original_img = cv2.imread("output.png", 0)

kernel = np.ones((20,20), np.uint8)
img_dilate = cv2.dilate(original_img, kernel)

img = cv2.resize(img_dilate, None, fx=0.8, fy=0.8,
                 interpolation=cv2.INTER_CUBIC)

# img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)  # 这里对最后一个参数使用了经验型的值
result = img.copy()


for line in lines:
    rho = line[0][0]  # 第一个元素是距离rho
    theta = line[0][1]  # 第二个元素是角度theta
    # print(rho)
    # print(theta)
    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
        pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
        # 该直线与最后一行的焦点
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
        cv2.line(result, pt1, pt2, (255))  # 绘制一条白线

        print(pt1,pt2)

    else:  # 水平直线
        pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
        # 该直线与最后一列的交点
        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
        cv2.line(result, pt1, pt2, (255), 1)  # 绘制一条直线

cv2.imshow('Canny', edges)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
