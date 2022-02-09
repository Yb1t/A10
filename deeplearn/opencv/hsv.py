import cv2
import matplotlib.pyplot as plt

def hex2hsv(color):
    r = int(color[:2], 16)
    g = int(color[2:4], 16)
    b = int(color[-2:], 16)
    h = 0
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    h, s, v=int(h), int(s*255), int(v*255)
    print("hue:%d saturation:%d value:%d" %(h,s,v))
    return h, s, v


img = cv2.imread('mb6.png')
blur = cv2.blur(img,(5,5))
blur0=cv2.medianBlur(blur,5)
blur1= cv2.GaussianBlur(blur0,(5,5),0)
blur2= cv2.bilateralFilter(blur1,9,75,75)
img = cv2.cvtColor(blur0, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

# 主板
# lower1 = (50,40,70)
# upper1 = (120,160,145)

lower1 = (20,40,40)
upper1 = (120,190,145)


lower2 = (0,0,0)
upper2 = (175,255,50)

# lower3 = (50,60,20)
# upper3 = (86,160,130)

# # 机箱
# lower1 = (20,20,50)
# upper1 = (70,40,140)
#
# lower2 = (0,0,70)
# upper2 = (80,35,160)


mask1 = cv2.inRange(hsv_img, lower1, upper1)
mask2 = cv2.inRange(hsv_img, lower2, upper2)
# mask3 = cv2.inRange(hsv_img, lower3, upper3)
mask = mask1 #+ mask2 #- mask3

result = cv2.bitwise_and(img, img, mask=mask)

plt.subplot(221)
plt.imshow(mask1,cmap="gray")
plt.subplot(222)
plt.imshow(mask2,cmap="gray")
plt.subplot(223)
plt.imshow(mask,cmap="gray")
plt.subplot(224)
plt.imshow(result)
plt.show()

cv2.imwrite("output.png", mask)