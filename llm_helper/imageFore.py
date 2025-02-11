# -*- coding: utf-8 -*-
# @Time    : 2024/7/29 15:07
# @Author  : zhanghaoxiang
# @File    : imageFore.py
# @Software: PyCharm
import cv2

# 读取图像
img = cv2.imread("D:\\temp\\zhongqi1\\zhongqi1_8_res\\parse_result\\8_1_table.jpg")

# 应用高斯模糊和拉普拉斯算子
blurred = cv2.GaussianBlur(img, (5, 5), 0)
sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

# 显示图像
cv2.imshow('Sharpened Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()