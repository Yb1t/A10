"""
数据集增强，opencv二值化图像获取主板绿色区域，保护已标记区域，网格找点覆盖螺丝等图像(png)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_INDEX = 88

DATASET_PATH = '/home/hao/Code/python/A10/datasets/board96/'  # 源数据集目录
train = DATASET_PATH + 'images/train/resized{}.jpg'.format(IMAGE_INDEX)  # 相对于目录的图片
label = DATASET_PATH + 'labels/train/resized{}.txt'.format(IMAGE_INDEX)  # 相对于目录的label
screw = '/home/hao/Code/python/A10/deeplearn/tools/img.png'

# 读取图片
screw_img = cv2.imread(screw, cv2.IMREAD_UNCHANGED)  # 带透明度读取
img = cv2.imread(train)
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

# 获取遮罩
mask1 = cv2.inRange(hsv_img, lower1, upper1)
mask2 = cv2.inRange(hsv_img, lower2, upper2)
mask = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))

# 腐蚀膨胀
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算
# mask = cv2.erode(mask, kernel)  # 腐蚀
# mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀

# plt绘图
plt.figure(figsize=(15, 15))
plt.suptitle('Dataset augmentation based on OpenCV')

plt.subplot(331), plt.title("source image")
plt.imshow(rgb_img)

plt.subplot(332), plt.title("green mask")
plt.imshow(mask1, cmap="gray")

plt.subplot(333), plt.title("desk green mask")
plt.imshow(mask2, cmap="gray")

plt.subplot(334), plt.title("result")
plt.imshow(cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_BGR2RGB))

cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("mask", 640, 480)
cv2.imshow('mask', mask)
# 获取主板遮罩结束
########################################################
# 计算已标注区域
data = np.loadtxt(label, dtype=float)  # [label][x%][y%][w%][h%]
percent_x = 0
percent_y = 0
exist = False
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
nx = ny = line_x = line_y = 0
if exist:
    nx = int(img.shape[1] / step)
    ny = int(img.shape[0] / step)
    line_x = nx - 1  # n-1条线
    line_y = ny - 1
else:
    print("There is NO target label index, check your label file:", label)
    print("exiting...")
    exit(1)
print("nx:{} ny:{} line_x:{} line_y:{}".format(nx, ny, line_x, line_y))

boxes = np.ones((line_y, line_x), dtype=np.uint8)
for item in data:
    x = nx * item[1]
    x_width = nx * item[3]
    y = ny * item[2]
    y_width = ny * item[4]
    lower = [min(round(x - x_width), line_x), min(round(y - y_width), line_y)]  # 不 /2 ，为原标记2倍大小
    upper = [min(round(x + x_width), line_x), min(round(y + y_width), line_y)]
    print("clearing:from {} to {}".format(lower, upper))
    for y in range(lower[1], upper[1]):
        for x in range(lower[0], upper[0]):
            print(x, y)
            boxes[y, x] = 0

# 建立以(1/nx,1/ny)为原点的分割线坐标系map01
map_raw = np.zeros((line_y, line_x), np.uint8)
near = 5  # 一个点的上下左右near个像素的4个点
for y in range(line_y):
    for x in range(line_x):
        # if step * x > img.shape[1] or step * y > img.shape[0]:
        #     break
        point_x = int(step * (x + 1))
        point_y = int(step * (y + 1))
        # 上下左右中有几个点在mask中
        map_raw[y, x] = 3 <= np.sum([mask[point_y, point_x] > 0,
                                     mask[point_y + near, point_x + near] > 0,
                                     mask[point_y - near, point_x - near] > 0,
                                     mask[point_y + near, point_x - near] > 0,
                                     mask[point_y - near, point_x + near] > 0]
                                    )

# 网格过滤器
grid_filter = np.zeros((line_y, line_x), np.uint8)
isOne = False
for y in range(line_y):
    for x in range(line_x):
        if isOne:
            grid_filter[y, x] = 1
        isOne = not isOne
    if line_x % 2 == 0:
        isOne = not isOne

# 计算可行图
map_available = map_raw * grid_filter * boxes

plt.subplot(335), plt.title("raw map")
plt.imshow(cv2.cvtColor(map_raw * 255, cv2.COLOR_GRAY2RGB))

plt.subplot(336), plt.title("labeled map")
plt.imshow(cv2.cvtColor(boxes * 255, cv2.COLOR_GRAY2RGB))

plt.subplot(337), plt.title("grid map")
plt.imshow(cv2.cvtColor(grid_filter * 255, cv2.COLOR_GRAY2RGB))

plt.subplot(338), plt.title("available map")
plt.imshow(cv2.cvtColor(map_available * 255, cv2.COLOR_GRAY2RGB))

points = []  # 可行点
baned_points = []  # 已经在标注区域内的点
padding = 1  # 图片四边留空padding个格
cover = np.zeros(img.shape, np.uint8)  # 覆盖层，覆盖在原图显示预览
for y in range(padding, line_y - padding):
    for x in range(padding, line_x - padding):
        pointx = (x + 1) * step  # 以(1/nx,1/ny)为原点
        pointy = (y + 1) * step
        cv2.line(cover, (pointx, 0), (pointx, img.shape[0]), [0, 0, 255], 2)
        cv2.line(cover, (0, pointy), (img.shape[1], pointy), [0, 0, 255], 2)
        cv2.putText(cover, str(x), (pointx, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
        cv2.putText(cover, str(y), (0, pointy), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
        if map_available[y, x] == 1:
            points.append([pointy, pointx])
            # print("add available point: x={}, y={}".format(pointx, pointy))
        elif boxes[y, x] == 0:
            baned_points.append([pointy, pointx])

print("available points:", len(points))

# 生成预览图
screw_size_x = int(step / 2)  # 插入图横向大小（像素）
screw_size_y = int(step / 2)  # 插入图纵向大小（像素）
count = 0  # 螺丝计数
for start in points:
    cover[(start[0] - screw_size_y):(start[0] + screw_size_y), (start[1] - screw_size_x):(start[1] + screw_size_x)] \
        = [0, 255, 0]
    count += 1
    cv2.putText(cover, str(count), (start[1], start[0]), cv2.FONT_HERSHEY_DUPLEX, .7, (255, 255, 255), 3)
for start in baned_points:
    cover[(start[0] - screw_size_y):(start[0] + screw_size_y), (start[1] - screw_size_x):(start[1] + screw_size_x)] \
        = [255, 0, 0]

preview_img = cv2.addWeighted(img, 1.0, cover, 0.5, 1)
cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
cv2.resizeWindow('preview', 640, 480)
cv2.imshow('preview', preview_img)

plt.subplot(339), plt.title("preview")
plt.imshow(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB))

# 加螺丝，生成最终结果图、对应的yolo数据集label.txt
result_img = img
labels = []
label_num = 6
for point in points:
    x_from = point[1] - step // 2
    y_from = point[0] - step // 2
    x_to = x_from + step
    y_to = y_from + step
    screw_img = cv2.resize(screw_img, (step, step))
    alpha = screw_img[:, :, 3] / 255
    for c in range(3):
        result_img[y_from:y_to, x_from:x_to, c] = (((1 - alpha) * result_img[y_from:y_to, x_from:x_to, c])
                                                   + (alpha * screw_img[:, :, c]))
    # [label][x%][y%][w%][h%]
    labels.append([label_num, point[1] / img_width, point[0] / img_height, img_width / step,
                   img_height / step])

for idx in range(len(labels)):
    for col in range(1, 3):
        labels[idx][col] = format(labels[idx][col], '.6f')  # 格式化成为6位小数
    for col in range(3, 5):
        labels[idx][col] = format(labels[idx][col] / 100, '.6f')

print("labels:{}...".format(labels[:2]))

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 640, 480)
cv2.imshow('result', result_img)

# 保存label
label_file_name = "{}.txt".format(IMAGE_INDEX)
print("saving labels: {}".format(label_file_name))
np.savetxt(label_file_name, X=labels, fmt='%s')

# 保存已处理图片
output_img_name = "{}.png".format(IMAGE_INDEX)
print("saving output image: {}".format(output_img_name))
cv2.imwrite(output_img_name, result_img)

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
