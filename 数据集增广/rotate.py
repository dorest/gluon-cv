# coding=utf-8
'''
先把图片变换，后缀+R

json文件坐标变换，注意变换前后seg的个数，后缀名改变，Imgname那项要改掉

'''
import os
import argparse
import imageio
import imgaug as ia
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import numpy as np
import json
import imgaug.augmenters as iaa
import random

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

def draw():
    image = imageio.imread('./flip/IMG_8038R.JPG')
    json_file = './flip/json/IMG_8038R.json'
    with open(json_file, 'rb') as fp:
        data = json.load(fp)
        all_point = []
        for shapes in data['shapes']:
            aaa = Polygon(shapes['points'])
            all_point.append(aaa)
        psoi = ia.PolygonsOnImage(all_point, shape=image.shape)
        aug = iaa.Sequential([
            iaa.Affine(scale=(1, 1), rotate=(0, 0), translate_percent=(0,0),
                       mode=["constant"], cval=0),  # 仿射变换

            # iaa.Fliplr(0.5), # 左右翻转
            # iaa.PerspectiveTransform((0.01, 0.1)), # 透视变换
            # iaa.AddToHueAndSaturation((-20, 20)),  #
            # iaa.LinearContrast((0.8, 1.2), per_channel=0.5),
            # iaa.Sometimes(0.75, iaa.Snowflakes())   # 高斯模糊
        ])
        image_aug, psoi_aug = aug(image=image, polygons=psoi)
        ia.imshow(psoi_aug.draw_on_image(image_aug, alpha_face=0.2, size_points=7))

# 旋转图片
def rotate_img(soudir, tardir, rate, json_dir, save_dir):
    pathdir = os.listdir(soudir)  # sourdir里不要有目录名
    filenum = len(pathdir)
    picknum = int(filenum*rate)
    sample = random.sample(pathdir, picknum)
    for name in sample:
        json_file = json_dir+name[:-4]+'.json'
        with open(json_file, 'rb') as fp:
            image = imageio.imread(soudir+name)
            data = json.load(fp)
            data['imagePath'] = data['imagePath'][:-4] + 'R.JPG'
            all_point = []
            for shapes in data['shapes']:
                aaa = Polygon(shapes['points'])
                all_point.append(aaa)
            psoi = ia.PolygonsOnImage(all_point, shape=image.shape)
            aug = iaa.Sequential([
                iaa.Affine(scale=(0.8, 1), rotate=(-10,10), translate_percent=(-0.1,0.1),
                           mode=["constant"], cval=0),  # 仿射变换

                # iaa.Fliplr(0.5), # 左右翻转
                # iaa.PerspectiveTransform((0.01, 0.1)), # 透视变换
                # iaa.AddToHueAndSaturation((-20, 20)),  #
                # iaa.LinearContrast((0.8, 1.2), per_channel=0.5),
                # iaa.Sometimes(0.75, iaa.Snowflakes())   # 高斯模糊
            ])
            image_aug, psoi_aug = aug(image=image, polygons=psoi)
            psoi_aug_removed = psoi_aug.remove_out_of_image(fully=True, partly=False)
            psoi_aug_removed_clipped = psoi_aug_removed.clip_out_of_image()
            assert len(psoi_aug.polygons) == len(psoi_aug_removed_clipped.polygons)
            for i,shapes in enumerate(data['shapes']):
                shapes['points'] = psoi_aug_removed_clipped.polygons[i].exterior.astype('int32').tolist()

            imageio.imwrite(tardir+name[:-4]+'R.JPG', image_aug)
            json.dump(data, open(save_dir + json_file[-13:-5] + 'R.json', 'w'), indent=4)


if __name__ == '__main__':
    args = parse_args()
    # rotate_img(args.imgdir, args.tardir, 1, args.jsondir, args.savedir)
    draw()
