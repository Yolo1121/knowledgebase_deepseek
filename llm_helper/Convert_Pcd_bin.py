# -*- coding: utf-8 -*-
# @Time    : 2024/9/10 13:20
# @Author  : zhanghaoxiang
# @File    : Convert_Pcd_bin.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Time : 2022/7/25 11:30
# @Author : JulyLi
# @File : pcd2bin.py

import numpy as np
import os
import argparse
from pypcd import pypcd
import csv
from tqdm import tqdm
def main():
    ## Add parser
    parser = argparse.ArgumentParser(description="Convert .pcd to .bin")
    parser.add_argument(
        "--pcd_path",
        help=".pcd file path.",
        type=str,
        default="D:\\cloudLabel\\labelCloud\pointclouds "
    )
    parser.add_argument(
        "--bin_path",
        help=".bin file path.",
        type=str,
        default="D:\\cloudLabel\\labelCloud\\pointclouds"
    )
    parser.add_argument(
        "--file_name",
        help="File name.",
        type=str,
        default=" 00001.bin"
    )
    args = parser.parse_args()

    ## Find all pcd files
    pcd_files = []
    ret= os.walk(args.pcd_path)
    print(args.pcd_path)
    for path, dir, files in os.walk("D:\\cloudLabel\\labelCloud\\pointclouds",topdown=False):
        for filename in files:
            # print(filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.pcd':
                pcd_files.append(path + "/" + filename)
   
    ## Sort pcd files by file name
    pcd_files.sort()
    print("Finish to load point clouds!")
    #print(pcd_files)

    ## Make bin_path directory
    try:
        if not (os.path.isdir(args.bin_path)):
            os.makedirs(os.path.join(args.bin_path))
    except OSError as e:
        # if e.errno != errno.EEXIST:
        #     print("Failed to create directory!!!!!")
            raise

    ## Generate csv meta file
    csv_file_path = os.path.join(args.bin_path, "meta.csv")
    csv_file = open(csv_file_path, "w")
    meta_file = csv.writer(
        csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    ## Write csv meta file header
    meta_file.writerow(
        [
            "pcd file name",
            "bin file name",
        ]
    )
    print("Finish to generate csv meta file")

    ## Converting Process
    print("Converting Start!")
    seq = 0
    for pcd_file in tqdm(pcd_files):
        ## Get pcd file
        pc = pypcd.PointCloud.from_path(pcd_file)

        ## Generate bin file name
        # bin_file_name = "{}_{:05d}.bin".format(args.file_name, seq)
        bin_file_name = "{:05d}.bin".format(seq)
        bin_file_path = os.path.join(args.bin_path, bin_file_name)

        ## Get data from pcd (x, y, z, intensity, ring, time)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc.pc_data['rgb'], dtype=np.float32)).astype(np.float32)
        # np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
        # np_t = (np.array(pc.pc_data['time'], dtype=np.float32)).astype(np.float32)

        ## Stack all data
        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))

        ## Save bin file
        points_32.tofile(bin_file_path)

        ## Write csv meta file
        meta_file.writerow(
            [os.path.split(pcd_file)[-1], bin_file_name]
        )

        seq = seq + 1


if __name__ == "__main__":
    main()

