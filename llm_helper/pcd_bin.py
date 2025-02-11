# -*- coding: utf-8 -*-
# @Time    : 2024/9/20 20:39
# @Author  : zhanghaoxiang
# @File    : pcd_bin.py
# @Software: PyCharm
import struct
point_cloud_data = []
# 读取PCD文件
with open("D:\\cloudLabel\\labelCloud\\pointclouds\\00001.txt", 'r') as f:
    xR:list=[]
    yR:list=[]
    zR:list=[]
    for line in f:
        # 解析每一行
        values = line.split()
        x, y, z = map(float, values)
        xR.append(x)
        yR.append(y)
        zR.append(z)
        #点云没有强度，此处利用0进行取代
        point_cloud_data.append((x, y, z,0))
    print(max(xR))
    print(min(xR))
    print(max(yR))
    print(min(yR))
    print(max(zR))
    print(min(zR))

with open("D:\\cloudLabel\\labelCloud\\pointclouds\\output.bin", "wb") as bin_file:
    for point in point_cloud_data:
        x, y, z, intensity = point
        bin_file.write(struct.pack("ffff", x, y, z, intensity))
print("PCD文件成功转换为BIN格式")


