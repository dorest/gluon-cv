# 数据集增广
介绍下做这个的背景：原始标注了500张图片，在想增广数据集的同时利用已标注的json文件。

dataug.py包含两种变换：
镜像翻转（flip）、颜色空间抖动（color jittering）

rotaty.py利用imgaug包实现：
旋转、缩放、平移操作
