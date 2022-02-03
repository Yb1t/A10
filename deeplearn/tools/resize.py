import os
from PIL import Image

base_dir = "./aaaaa/"
filename = os.listdir(base_dir)  # 原图片集所在目录
new_dir = "./bbbbb/"  # 缩放后的图片存放的路径
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
new_longest = 640  # 设置新图像最长边像素数

for img in filename:  # 遍历文件夹中的图片
    image = Image.open(base_dir + img)
    width = image.size[0]  # 原图宽
    height = image.size[1]  # 原图高
    # 以下代码实现按设置好的最长边像素数缩放图片，并保持宽高比不变
    if width <= height:
        longest = width
        new_height = int(height * new_longest / width)
        out = image.resize((new_longest, new_height), Image.ANTIALIAS)
    else:
        longest = height
        new_width = int(width * new_longest / height)
        out = image.resize((new_width, new_longest), Image.ANTIALIAS)
    out.save(new_dir + img)  # 保存新图片

