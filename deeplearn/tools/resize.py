import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Image resizer')
parser.add_argument("--base-dir", type=str, default="/home/hao/Documents/比赛/服创13/服创大赛数据集/等于/")
parser.add_argument("--new-dir", type=str, default="/home/hao/Downloads/output/")
parser.add_argument("--index", type=int, default=92)
parser.add_argument("--name", type=str, default="resized")
parser.add_argument("--ext", type=str, default="jpg")
args = parser.parse_args()

base_dir = args.base_dir
filename = os.listdir(base_dir)  # 原图片集所在目录
new_dir = args.new_dir  # 缩放后的图片存放的路径
new_name = args.name
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
new_longest = 1280  # 设置新图像最长边像素数
num = args.index  # 已处理图片数
ext = args.ext

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
    output = "{}{}{}.{}".format(new_dir, new_name, num, ext)
    out.save(output)  # 保存新图片
    print(output)
    num += 1
