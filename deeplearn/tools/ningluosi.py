"""
数据集增强，opencv二值化图像获取主板绿色区域，网格找点覆盖螺丝等图像(png)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_INDEX = 36

DATASET_PATH = '/home/hao/Code/python/A10/datasets/board96/'
train_image = DATASET_PATH + 'images/train/resized{}.jpg'.format(IMAGE_INDEX)
label = DATASET_PATH + 'labels/train/resized{}.txt'.format(IMAGE_INDEX)

img = cv2.imread(train_image)
print("img shape:{}".format(img.shape))
img_width = img.shape[1]
img_height = img.shape[0]
# blur = cv2.blur(img,(5,5))
# blur0=cv2.medianBlur(blur,5)
# blur1= cv2.GaussianBlur(blur0,(5,5),0)
# blur2= cv2.bilateralFilter(blur1,9,75,75)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

# 主板绿
lower1 = (30, 50, 40)
upper1 = (80, 190, 255)

# 桌面绿
lower2 = (60, 100, 40)
upper2 = (90, 200, 150)

mask1 = cv2.inRange(hsv_img, lower1, upper1)
mask2 = cv2.inRange(hsv_img, lower2, upper2)
mask = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))

# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算
# mask = cv2.erode(mask, kernel)  # 腐蚀
# mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀

plt.subplot(221)
plt.imshow(img)

plt.subplot(222)
plt.imshow(mask2, cmap="gray")

plt.subplot(223)
plt.imshow(mask1, cmap="gray")

plt.subplot(224)
plt.imshow(mask, cmap="gray")

plt.show()

cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mask", 640, 480)
cv2.imshow('mask', mask)
# cv2.imwrite("01.png", mask)

########################################################

# img = cv2.imread(train_image)
data = np.loadtxt(label, dtype=float)  # [label][x%][y%][w%][h%]
percent_x = 0
percent_y = 0
exist = False
boxes = []
print("calculating box size with item(s):")
for item in data:
    # 主板标注大小（百分比）
    if item[0] == 6.0 or item[0] == 0 or item[0] == 1.0:
        exist = True
        print(item)
        percent_x = (item[3] + percent_x) / 2
        percent_y = (item[4] + percent_y) / 2
step = int((percent_x + percent_y) / 2 * (img.shape[1] + img.shape[0]) / 2)
print("step:", step)
nx = int(img.shape[1] / step) if exist else 0
ny = int(img.shape[0] / step) if exist else 0
print("nx:{} ny:{}".format(nx, ny), round(nx*data[0, 1]))

# 建立以(1/nx,1/ny)为原点的分割线坐标系map01
line_x = nx-1
line_y = ny-1
map01 = np.zeros((line_x, line_y), np.uint8)  # n-1条线
near = 5
for y in range(line_y):
    for x in range(line_x):
        # if step * x > img.shape[1] or step * y > img.shape[0]:
        #     break
        point_x = int(step * (x+1))
        point_y = int(step * (y+1))
        map01[x, y] = 3 <= np.sum([mask[point_y, point_x] > 0,
                                   mask[point_y + near, point_x + near] > 0,
                                   mask[point_y - near, point_x - near] > 0,
                                   mask[point_y + near, point_x - near] > 0,
                                   mask[point_y - near, point_x + near] > 0]
                                  )

shai = np.zeros((line_x, line_y), np.uint8)
isOne = False
for y in range(line_y):
    for x in range(line_x):
        if isOne:
            shai[x, y] = 1
        isOne = not isOne
    if line_x % 2 == 0:
        isOne = not isOne

result = map01 * shai

points = []
padding = 0
for y in range(padding, line_y - padding):
    for x in range(padding, line_x - padding):
        pointx = (x+1) * step  # 以(1/nx,1/ny)为原点
        pointy = (y+1) * step
        cv2.line(img, (pointx, 0), (pointx, img.shape[0]), [0, 0, 255], 2)
        cv2.line(img, (0, pointy), (img.shape[1], pointy), [0, 0, 255], 2)
        if result[x, y] == 1:
            points.append([pointy, pointx])

print("points:", len(points))

screw_size_x = int(step / 2)
screw_size_y = int(step / 2)
for start in points:
    img[(start[0] - screw_size_x):(start[0] + screw_size_y), (start[1] - screw_size_x):(start[1] + screw_size_y)] \
        = [255, 0, 0]
cv2.namedWindow("1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("1", 640, 480)
cv2.imshow("1", img)
cv2.waitKey()
cv2.destroyAllWindows()
