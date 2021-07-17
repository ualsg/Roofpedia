import glob
import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image
import random
import cv2 
import matplotlib.pyplot as plt 
from src.colors import make_palette
os.getcwd()

def load_img(target_path, source_path):
    files_target = glob.glob(target_path + '\*\*\*.png', recursive=True)
    files_source = glob.glob(source_path + '\*\*\*.png', recursive=True)
    # files_source = []
    # for i in files_target:
    #     files_source.append(i.replace('Target', 'Black')) 
    print(str(len(files_target)) + ' target files found')
    print(str(len(files_source)) + ' source files found')
    print('Source file example :' + str(files_target[1]))
    print('Source file example :' + str(files_source[1]))
    return files_target, files_source

def remove_blank_tiles(files_target):
    # find all blank tiles
    rm_list = []
    for file in files_target:
        img = cv2.imread(file)
        if np.unique(img, return_counts=True)[1][0] == 196608:
            rm_list.append(file)

    # get list of all blank tiles in all folders
    if len(rm_list) != 0:
        rm_sat_image = []
        for i in rm_list:
            rm_sat_image.append(i.replace('labels', 'images')) 

    # remove all blank tiles
    all_list = zip(rm_list, rm_sat_image)
    for f in all_list:
        os.remove(f[0])
        os.remove(f[1])

# train test val split
def train_test_split(file_list, train_size = 0.7):
    random.Random(123).shuffle(file_list)
    train_stop = int(len(file_list)*train_size)
    test_stop = int((len(file_list) - train_stop)/2)
    train_data = file_list[:train_stop]
    test_data = file_list[train_stop: train_stop + test_stop]
    val_data = file_list[train_stop + test_stop:]
    return train_data, test_data, val_data


if __name__ == "__main__":
    target_path = 'labels'
    source_path = 'images'
    files_target, files_source = load_img(target_path, source_path)

    # remove blank imgs
    remove_blank_tiles(files_target)

    target_path = 'labels'
    source_path = 'images'
    files_target, files_source = load_img(target_path, source_path)