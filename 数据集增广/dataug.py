'''
|——json总文件
|——原始图
|——随机翻转图
    |——翻转图的json文件夹
    |——翻图1
    |——翻图2
    |:

'''

import os
import argparse
import glob
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import random
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Test with Mask RCNN networks.')
    # wjh
    parser.add_argument('--imgdir', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--tardir', type=str, default='',
                        help='变换后的图像路径')
    parser.add_argument('--savedir', type=str, default='',
                        help='保存变换json的路径')
    parser.add_argument('--jsondir', type=str, default='',
                        help='总json文件的路径')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')
    args = parser.parse_args()
    return args

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(3, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(3, 12) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度增强因子为0.0将产生黑色图像；为1.0将保持原始图像
    random_factor = np.random.randint(3, 15) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(8, 21) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度random_factor)


# 翻转图片
def flip_img(soudir, tardir, rate):
    pathdir = os.listdir(soudir)  # sourdir里不要有目录名
    filenum = len(pathdir)
    picknum = int(filenum*rate)
    sample = random.sample(pathdir, picknum)
    for name in sample:
        img = cv.imread(soudir+name)
        horizontal = cv.flip(img, 1, dst=None)  # 水平翻转
        im = Image.fromarray(cv.cvtColor(horizontal, cv.COLOR_BGR2RGB))
        im = randomColor(im)
        im.save(tardir+name[:-4]+'F.JPG')

# 翻转图片后的json文件
def flip_json(img_dir, json_dir, save_dir):
    all_json_files = glob.glob(json_dir + '*.json')
    sample = os.listdir(img_dir)
    sample.remove('json')
    json_files = []
    for i in all_json_files:
        for j in sample:
            if j[:-5] in i:
                json_files.append(i)
    for num, json_file in enumerate(json_files):

        with open(json_file, 'rb') as fp:
            data = json.load(fp)  # 加载json文件
            data['imagePath'] = data['imagePath'][:-4] + 'F.JPG'
            imgdir = img_dir+data['imagePath'][3:]
            img = cv.imread(imgdir, 0)
            height, width = img.shape[:2]
            for shapes in data['shapes']:
                for point in shapes['points']:
                    point[0] = width - point[0]

            json.dump(data, open(save_dir+json_file[-13:-5]+'F.json', 'w'), indent=4)

# 改变颜色空间
def color_img():
    pass


if __name__ == '__main__':
    args = parse_args()
    img_list = glob.glob(args.imgdir+'*.JPG')

    flip_img(args.imgdir, args.tardir, 0.7)
    flip_json(args.tardir, args.jsondir, args.savedir)
