"""
数据集增强，opencv二值化图像获取主板绿色区域，保护已标记区域，网格找点覆盖透视变换后的螺丝等图像（3通道）
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


# 获取主板遮罩
def get_mask(img):
    print("img shape:{}".format(img.shape))
    # #模糊
    # blur = cv2.blur(img,(5,5))
    # blur0=cv2.medianBlur(blur,5)
    # blur1= cv2.GaussianBlur(blur0,(5,5),0)
    # blur2= cv2.bilateralFilter(blur1,9,75,75)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
    if IS_DRAW_PLT:
        plt.figure(figsize=(15, 15))
        plt.suptitle('Dataset augmentation based on OpenCV')
        plt.subplot(331), plt.title("source image")
        plt.imshow(cv2.cvtColor(SourceImg, cv2.COLOR_BGR2RGB))
        plt.subplot(332), plt.title("green mask")
        plt.imshow(mask1, cmap="gray")
        plt.subplot(333), plt.title("desk green mask")
        plt.imshow(mask2, cmap="gray")
        plt.subplot(334), plt.title("total mask")
        plt.imshow(cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_BGR2RGB))

    return mask


# 计算步长
def get_step(label_data):
    percent_x = 0
    percent_y = 0
    exist = False
    print("calculating box size with item(s):")
    # label data: [label][x%][y%][w%][h%]
    for item in label_data:
        # 计算平均标注大小（百分比）
        if item[0] == 6.0 or item[0] == 0 or item[0] == 1.0:
            exist = True
            print(item)
            percent_x = (item[3] + percent_x) / 2
            percent_y = (item[4] + percent_y) / 2
    if not exist:
        print("There is NO target label index, check your label file:", LabelPath)
        print("exiting...")
        exit(1)
    step = int((percent_x + percent_y) / 2 * (SourceImg.shape[1] + SourceImg.shape[0]) / 2)
    print("step:", step)
    return step


# 计算横纵各有多少格、多少条线分割
def get_n_line_num(step):
    nx = int(SourceImg.shape[1] / step)
    ny = int(SourceImg.shape[0] / step)
    line_x = nx - 1  # n-1条线
    line_y = ny - 1
    print("nx:{} ny:{} line_x:{} line_y:{}".format(nx, ny, line_x, line_y))
    return nx, ny, line_x, line_y


# 计算已标注区域
def get_labeled_filter(label_data, nx, ny, line_x, line_y):
    # 计算filter
    labeled_filter = np.ones((line_y, line_x), dtype=np.uint8)
    for item in label_data:
        x = nx * item[1]
        x_width = nx * item[3]
        y = ny * item[2]
        y_width = ny * item[4]
        lower = [min(round(x - x_width), line_x), min(round(y - y_width), line_y)]  # 不 /2 ，为原标记2倍大小
        upper = [min(round(x + x_width), line_x), min(round(y + y_width), line_y)]
        print("excluding labeled area:from {} to {}".format(lower, upper))
        for y in range(lower[1], upper[1]):
            for x in range(lower[0], upper[0]):
                # print(x, y)
                labeled_filter[y, x] = 0
    return labeled_filter


# 根据mask生成初步 map_raw
def get_raw_map(mask):
    # 建立以(1/nx,1/ny)为原点的分割线坐标系 map_raw
    map_raw = np.zeros((LineNumY, LineNumX), np.uint8)
    near = 5  # 一个点的上下左右near个像素的4个点
    for y in range(LineNumY):
        for x in range(LineNumX):
            # if step * x > img.shape[1] or step * y > img.shape[0]:
            #     break
            point_x = int(Step * (x + 1))
            point_y = int(Step * (y + 1))
            # 上下左右中有几个点在mask中
            map_raw[y, x] = 3 <= np.sum([mask[point_y, point_x] > 0,
                                         mask[point_y + near, point_x + near] > 0,
                                         mask[point_y - near, point_x - near] > 0,
                                         mask[point_y + near, point_x - near] > 0,
                                         mask[point_y - near, point_x + near] > 0]
                                        )
    return map_raw


# 生成网格过滤器 map_grid
def get_grid_filter(num_x, num_y):
    # 网格过滤器
    grid_filter = np.zeros((num_y, num_x), np.uint8)
    is_one = False
    for y in range(LineNumY):
        for x in range(LineNumX):
            if is_one:
                grid_filter[y, x] = 1
            is_one = not is_one
        if LineNumX % 2 == 0:
            is_one = not is_one
    return grid_filter


# 计算可行图
def get_available_map(raw_map, grid_filter, labeled_filter):
    available_map = raw_map * grid_filter * labeled_filter
    if IS_DRAW_PLT:
        plt.subplot(335), plt.title("raw map")
        plt.imshow(cv2.cvtColor(raw_map * 255, cv2.COLOR_GRAY2RGB))
        plt.subplot(336), plt.title("labeled map")
        plt.imshow(cv2.cvtColor(labeled_filter * 255, cv2.COLOR_GRAY2RGB))
        plt.subplot(337), plt.title("grid map")
        plt.imshow(cv2.cvtColor(grid_filter * 255, cv2.COLOR_GRAY2RGB))
        plt.subplot(338), plt.title("available map")
        plt.imshow(cv2.cvtColor(available_map * 255, cv2.COLOR_GRAY2RGB))
    return available_map


# 计算可行点
def get_points(available_map, labeled_filter):
    points = []  # 可行点
    baned_points = []  # 已经在标注区域内的点
    padding = 1  # 图片四边留空padding个格
    cover = np.zeros(SourceImg.shape, np.uint8)  # 覆盖层，覆盖在原图显示预览
    for y in range(padding, LineNumY - padding):
        for x in range(padding, LineNumX - padding):
            pointx = (x + 1) * Step  # 以(1/nx,1/ny)为原点
            pointy = (y + 1) * Step
            cv2.line(cover, (pointx, 0), (pointx, SourceImg.shape[0]), [0, 0, 255], 2)
            cv2.line(cover, (0, pointy), (SourceImg.shape[1], pointy), [0, 0, 255], 2)
            cv2.putText(cover, str(x), (pointx, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
            cv2.putText(cover, str(y), (0, pointy), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
            if available_map[y, x] == 1:
                points.append([pointy, pointx])
                # print("add available point: x={}, y={}".format(pointx, pointy))
            elif labeled_filter[y, x] == 0:
                baned_points.append([pointy, pointx])
    print("available point(s):", len(points))
    return points, baned_points


# 生成预览图
def get_preview(points, baned_points, source_img):
    cover = np.zeros(source_img.shape, np.uint8)
    screw_w = int(Step / 2)  # 插入图横向大小（像素）
    screw_h = int(Step / 2)  # 插入图纵向大小（像素）
    count = 0  # 螺丝计数
    for start in points:
        cover[(start[0] - screw_h):(start[0] + screw_h), (start[1] - screw_w):(start[1] + screw_w)] = [0, 255, 0]
        count += 1
        cv2.putText(cover, str(count), (start[1], start[0]), cv2.FONT_HERSHEY_DUPLEX, .7, (255, 255, 255), 3)
    for start in baned_points:
        cover[(start[0] - screw_h):(start[0] + screw_h), (start[1] - screw_w):(start[1] + screw_w)] = [255, 0, 0]
    preview_img = cv2.addWeighted(source_img, 1.0, cover, 0.5, 1)
    if IS_DRAW_PLT:
        plt.subplot(339), plt.title("preview")
        plt.imshow(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB))
    return preview_img


# 由坐标生成透视变化后的螺丝
def get_screw(screw_img, x, y):
    w = screw_img.shape[1]
    h = screw_img.shape[0]

    hsv_img = cv2.cvtColor(screw_img, cv2.COLOR_BGR2HSV)
    # 主板绿
    lower1 = (30, 50, 40)
    upper1 = (80, 190, 255)
    alpha_channel = cv2.bitwise_not(cv2.inRange(hsv_img, lower1, upper1))
    b_channel, g_channel, r_channel = cv2.split(screw_img)
    screw_img_rgba = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    # 原图中卡片在左上、右上、左下、右下的四个角点
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    perc_y = (y - SourceImgHeight / 2) / SourceImgHeight  # y到水平中轴距离占图片竖轴百分比，-0.5~0.5
    top_span = w * (.5 - perc_y) * .5 * .2  # 原图片的左上角(0,0)移到(top_span,0)，以此类推
    bottom_span = w * (.5 + perc_y) * .5 * .2
    # 变换后分别四个点
    change4y = np.float32([[top_span, 0], [w - top_span, 0], [bottom_span, h], [w - bottom_span, h]])  # up
    # print("top_span{}/{},bottom_span{}/{}".format(top_span, w, bottom_span, w))

    perc_x = (x - SourceImgWidth / 2) / SourceImgWidth  # x到竖直中轴距离占图片横轴百分比，-0.5~0.5
    left_span = h * (.5 - perc_x) * .5 * .2
    right_span = h * (.5 + perc_x) * .5 * .2
    change4x = np.float32([[0, left_span], [w, right_span], [0, w - left_span], [w, h - right_span]])  # up
    # print("left_span{}/{},right_span{}/{}".format(left_span, h, right_span, h))

    # 生成透视变换矩阵
    metric_y = cv2.getPerspectiveTransform(pts1, change4y)
    metric_x = cv2.getPerspectiveTransform(pts1, change4x)
    # 进行透视变换，规定目标图像大小
    res_y = cv2.warpPerspective(screw_img_rgba, metric_y, (w, h))
    res_y_x = cv2.warpPerspective(res_y, metric_x, (w, h))

    # cv2.imshow('img {} {}'.format(x, y), screw_img_rgba)
    # cv2.imshow('res_y {} {}'.format(x, y), res_y)
    # cv2.imshow('res_y_x {} {}'.format(x, y), res_y_x)
    # cv2.waitKey()

    return res_y_x


# 加螺丝，生成最终结果图、对应的新yolo数据集label
def get_result(available_points):
    result_img = SourceImg
    new_labels = []
    label_num = 6
    for point in available_points:
        x_from = point[1] - Step // 2
        y_from = point[0] - Step // 2
        x_to = x_from + Step
        y_to = y_from + Step
        screw_alpha = get_screw(ScrewImg, point[1], point[0])
        screw_img = cv2.resize(screw_alpha, (Step, Step))
        alpha = screw_img[:, :, 3] / 255
        # cv2.imshow('t',alpha)
        # cv2.waitKey()
        for c in range(3):
            result_img[y_from:y_to, x_from:x_to, c] = (((1 - alpha) * result_img[y_from:y_to, x_from:x_to, c])
                                                       + (alpha * screw_img[:, :, c]))
        # [label][x%][y%][w%][h%]
        new_labels.append([label_num,
                           point[1] / SourceImgWidth / 100,
                           point[0] / SourceImgHeight / 100,
                           SourceImgWidth / Step / 100,
                           SourceImgHeight / Step / 100
                           ])
    print("new label(s):", len(new_labels))
    return result_img, new_labels


# 保存结果
def save(result_img, new_labels):
    new_img_dir = DATASET_PATH + 'images_augmented/train/'
    new_label_dir = DATASET_PATH + 'labels_augmented/train/'
    if not os.path.exists(new_img_dir):
        os.makedirs(new_img_dir)
    if not os.path.exists(new_label_dir):
        os.makedirs(new_label_dir)
    # 保存label
    labels_save = LabelData.tolist() + new_labels
    for idx in range(len(labels_save)):
        labels_save[idx][0] = int(labels_save[idx][0])
        for col in range(1, 5):
            labels_save[idx][col] = format(labels_save[idx][col], '.6f')  # 格式化成为6位小数
    label_file_name = "{}.txt".format(NameNoExt)
    print("saving labels:", label_file_name)
    np.savetxt(new_label_dir + label_file_name, X=labels_save, fmt='%s')
    print("{} labels saved: {} old label(s), {} new label(s)".format(
        len(LabelData) + len(new_labels), len(LabelData), len(new_labels)))
    # 保存已处理图片
    show_img_in_window('', result_img)
    # output_img_name = "{}.jpg".format(NameNoExt)
    # print("saving output image: {}".format(output_img_name))
    # cv2.imwrite(new_img_dir + output_img_name, result_img)
    # print("result image saved:", new_img_dir + output_img_name)


# 在窗口中显示图片
def show_img_in_window(title, img):
    title = "[{}] {}".format(NameNoExt, title)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 640, 480)
    cv2.imshow(title, img)


if __name__ == '__main__':
    IS_DRAW_PLT = False
    IS_SHOW_RESULT = False
    IS_SHOW_MASK = False
    IS_PREVIEW = False
    IS_SAVE = True
    Screw = 'screw.png'
    DATASET_PATH = '/home/hao/Downloads/dataset/'  # 源数据集目录
    Names = os.listdir(DATASET_PATH + 'images/train/')
    NumProcessed = 0
    for name in Names:
        start_time = perf_counter()
        NumProcessed += 1
        print("processing ({}/{}) ...".format(NumProcessed, len(Names)))
        NameNoExt = name.split('.')[0]
        Ext = name.split('.')[-1]
        ImagePath = DATASET_PATH + 'images/train/' + name
        LabelPath = DATASET_PATH + 'labels/train/' + NameNoExt + '.txt'

        SourceImg = cv2.imread(ImagePath)  # 原图
        ScrewImg = cv2.imread(Screw, cv2.IMREAD_UNCHANGED)  # 螺丝图，带透明度
        SourceImgWidth = SourceImg.shape[1]
        SourceImgHeight = SourceImg.shape[0]

        LabelData = np.loadtxt(LabelPath, dtype=float)
        Mask = get_mask(SourceImg)
        if IS_SHOW_MASK:
            show_img_in_window('mask', Mask)
        Step = get_step(LabelData)
        NX, NY, LineNumX, LineNumY = get_n_line_num(Step)
        MapRaw = get_raw_map(Mask)
        FilterLabeled = get_labeled_filter(LabelData, NX, NY, LineNumX, LineNumY)
        FilterGrid = get_grid_filter(LineNumX, LineNumY)
        MapAvailable = get_available_map(MapRaw, FilterGrid, FilterLabeled)
        PointsAvailable, PointsBaned = get_points(MapAvailable, FilterLabeled)
        if IS_PREVIEW:
            Preview = get_preview(PointsAvailable, PointsBaned, SourceImg)
            show_img_in_window('preview', Preview)
        ImgResult, LabelsNew = get_result(PointsAvailable)

        if IS_SAVE:
            save(ImgResult, LabelsNew)

        print("consume time:", perf_counter() - start_time)

        if IS_DRAW_PLT:
            plt.show()

        if IS_SHOW_RESULT:
            show_img_in_window('result', ImgResult)
        cv2.waitKey()
        cv2.destroyAllWindows()

        print("---" * 20)

    exit(0)
