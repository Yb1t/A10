"""
找主板边，可进一步获取主板4个角坐标，作用于数据集增广中透视变换、提取高清主板图片
（WIP）通过颜色范围获取主板遮罩，加以腐蚀膨胀获得整块区域，由霍夫变换得出所有可能的边缘直线，再由角度过滤得到接近垂直和水平的直线
方式缺点：极其不稳定，受环境因素影响大
"""
import cv2
import numpy as np

ACC = 130
img = cv2.imread('/home/hao/Code/python/A10/datasets/board96/images/train/resized64.jpg')
blur = cv2.blur(img, (5, 5))
blur0 = cv2.medianBlur(blur, 5)
blur1 = cv2.GaussianBlur(blur0, (5, 5), 0)
# blur2= cv2.bilateralFilter(blur1,9,75,75)
img = cv2.cvtColor(blur1, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# 主板绿
lower1 = (30, 50, 40)
upper1 = (80, 190, 255)

# 桌面绿
lower2 = (60, 100, 40)
upper2 = (90, 200, 150)

# 获取遮罩
mask1 = cv2.inRange(hsv_img, lower1, upper1)
mask2 = cv2.inRange(hsv_img, lower2, upper2)

mask = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))

cv2.namedWindow('m', cv2.WINDOW_NORMAL)
cv2.resizeWindow('m', 640, 480)
cv2.imshow('m', mask)

# 腐蚀膨胀
kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算
mask = cv2.erode(mask, kernel, iterations=1)  # 腐蚀
kernel = np.ones((8, 8), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=10)  # 膨胀

# res = cv2.bitwise_and(img, img, mask=mask)

img = cv2.resize(mask, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

# img = cv2.GaussianBlur(img, (3, 3), 0)
edges = cv2.Canny(img, 50, 150, apertureSize=3)  # 边缘检测
lines = cv2.HoughLines(edges, 1, np.pi / 180, ACC)
result = img.copy()

for line in lines:
    rho = line[0][0]  # 第一个元素是距离rho
    theta = line[0][1]  # 第二个元素是角度theta
    # print(rho, theta)
    if (theta < (np.pi * .1)) and (theta > (np.pi * -.1)):  # 垂直直线
        pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
        # 该直线与最后一行的焦点
        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
        cv2.line(result, pt1, pt2, 255, 2)  # 绘制一条白线
    elif (theta > (np.pi / 2. - np.pi * .1)) and (theta < (np.pi / 2. + np.pi * .1)):  # 水平直线
        pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
        # 该直线与最后一列的交点
        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
        cv2.line(result, pt1, pt2, 255, 2)  # 绘制一条直线

cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Canny', 640, 480)
cv2.imshow('Canny', edges)

cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 640, 480)
cv2.imshow('Result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
