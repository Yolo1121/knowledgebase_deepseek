# -*- coding: utf-8 -*-
# @Time    : 2024/6/26 11:36
# @Author  : zhanghaoxiang
# @File    : text_ocr.py
# @Software: PyCharm
from PIL import Image,ImageOps
from paddleocr import PPStructure, save_structure_res, PaddleOCR
import re
import math
import numpy as np

class TextOCR(object):
    def __init__(self,img_path:str):
        self.img_path = img_path
    def text_ocr(self):
        '''
        利用ocr识别文本图片
        :return:
        '''
        try:
            img = Image.open(self.img_path)
            tem=img.filename
            end = "\\"
            string2 = tem[tem.rfind(end):]
            pat2 = '\\\(.+?\.jpg)'
            ret=re.compile(pat2).findall(string2)
            if(ret[0].split('_')[-1].split('.')[0]== "text" or ret[0].split('_')[-1].split('.')[0]== "reference" or ret[0].split('_')[-1].split('.')[0]== "figurecaption" or ret[0].split('_')[-1].split('.')[0]== "tablecaption"):
                ocr1 = PaddleOCR(use_gpu=False, lang='ch')
                #修正图像大小，将图像伸缩到320
                border=[0,0]
                h, w = img.size[1], img.size[0]
                transform_size = 640
                if w < transform_size or h < transform_size:
                    if h < transform_size:
                        border[0]=int((transform_size - h) / 2.0+0.5)
                    if w < transform_size:
                        border[1] =int((transform_size - w) / 2.0+0.5)
                new_img=ImageOps.expand(img, border=(border[1],border[0], border[1],border[0]),fill=(255, 255, 255))
                #img.show(title='原始图像')
                #new_img.show("扩充图像")
                imr_arrary=np.array(new_img)
                ret=ocr1.ocr(imr_arrary)
                print(self.img_path)
                return 400, ret
            else:
                print('异常'+self.img_path)
                return 401, None

        except Exception as e:
            return 402, e
    def retPrase(self,result):
        try:
            txt = ""
            temp = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    # print(line[1][0])
                    temp.append(line[1][0])
            return txt.join(temp)
        except Exception as e:
            pass


'''
a=TextOCR("D:\\temp\\Blip\\Blip_4_res\\parse_result\\4_9_figurecaption.jpg")
code,ret1=a.text_ocr()
ret=a.retPrase(ret1)
print(ret)
'''
