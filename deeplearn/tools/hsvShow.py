""" 在HSV中的图像生成彩色3D散点图 """
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

img = cv2.imread('img.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
fig = plt.figure(figsize=(8, 6), dpi=80)
axis = fig.add_subplot(1, 1, 1, projection="3d")

# 像素颜色设置
pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
# 归一化
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
# 转换成list
pixel_colors = norm(pixel_colors).tolist()

# 显示三维散点图
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker='.')
axis.set_xlabel("hue")
axis.set_ylabel("saturation")
axis.set_zlabel("value")
plt.show()
