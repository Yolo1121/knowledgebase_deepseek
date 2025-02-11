# -*- coding: utf-8 -*-
# @Time    : 2024/8/1 15:15
# @Author  : zhanghaoxiang
# @File    : image_recover.py
# @Software: PyCharm
import os
import cv2
from paddleocr import PPStructure,save_structure_res
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

# 中文测试图
table_engine = PPStructure(recovery=True)
# 英文测试图
# table_engine = PPStructure(recovery=True, lang='en')

save_folder = './output'
img_path = 'D://temp//zhongqi1//zhongqi1_8_res//zhongqi1_8.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

h, w, _ = img.shape
res = sorted_layout_boxes(result, w)
convert_info_docx(img, res, save_folder, os.path.basename(img_path).split('.')[0])