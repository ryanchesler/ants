import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
import os
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--obj_path")
parser.add_argument("--target_path")
args = parser.parse_args()
file_list = glob.glob(f"{args.obj_path}/*/*_points.obj")
os.makedirs(args.target_path)
os.makedirs("segment_arrays", exist_ok = True)
for file in tqdm(file_list):
    z_counter = {}
    try:
        all_points = []
        with open(file, "r") as f:
            points = f.readlines()
            for line in points:
                info = line.split()
                if len(info) != 4 or info[0] != "v":
                    continue
                x, y, z = int((float(info[1]))), int((float(info[2]))), int((float(info[3])))
                if z in z_counter:
                    z_counter[z] += 1
                else:
                    z_counter[z] = 1
                all_points.append([x, y, z])
        all_points = np.array(all_points)
        
        all_points = all_points.reshape(-1, z_counter[z-1], 3)
        print(file, all_points.shape)
        
        np.save(f"segment_arrays/{file.split('/')[-1].split('_')[0]}.npy", all_points)
    except:
        print(file)
