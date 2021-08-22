import glob
import numpy as np
import os
import shutil
from PIL import Image
import random
import cv2 
from src.colors import make_palette
os.getcwd()

def load_img(target_path, source_path):
    files_target = glob.glob(target_path + '\*\*\*.png', recursive=True)
    files_source = glob.glob(source_path + '\*\*\*.png', recursive=True)
    print(str(len(files_target)) + ' target files found')
    print(str(len(files_source)) + ' source files found')
    return files_target, files_source

def remove_blank_tiles(files_target):
    # find all blank tiles
    rm_list = []
    for file in files_target:
        img = cv2.imread(file)
        if np.unique(img, return_counts=True)[1][0] == 196608:
            rm_list.append(file)

    # get list of all blank tiles in all folders
    rm_sat_image = []
    if len(rm_list) != 0:
        for i in rm_list:
            rm_sat_image.append(i.replace('labels', 'images')) 

        # remove all blank tiles
    all_list = zip(rm_list, rm_sat_image)
    for f in all_list:
        os.remove(f[0])
        os.remove(f[1])
    print(str(len(rm_list)) + " blank images removed")

def convert_mask(mask_list):
    for i in mask_list:
        img = Image.open(i)
        thresh = 255
        fn = lambda x : 255 if x < thresh else 0
        out = img.convert('P').point(fn, mode='1')
        out = out.convert('P')
        palette = make_palette("dark", "light")
        out.putpalette(palette)
        out.save(i)
    print("Masks converted to 1bit labels, please check for correctness")
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
    target_path = 'dataset/labels'
    source_path = 'dataset/images'
    files_target, files_source = load_img(target_path, source_path)
    remove_blank_tiles(files_target)
    print("reloading trimmed data")
    files_target, files_source = load_img(target_path, source_path)
    convert_mask(files_target)

    train_data, test_data, val_data = train_test_split(files_target, train_size = 0.7)
    train_data_img = []
    test_data_img =[]
    val_data_img =[]
    
    output_folder = 'dataset'
    
    for i in train_data:
        if not os.path.exists(output_folder +'/training/labels/'):
            os.makedirs(output_folder +'/training/labels/')
        dest = output_folder +'/training/labels/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in test_data:
        if not os.path.exists(output_folder +'/evaluation/labels/'):
            os.makedirs(output_folder +'/evaluation/labels/')
        dest = output_folder +'/evaluation/labels/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in val_data:
        if not os.path.exists(output_folder +'/validation/labels/'):
            os.makedirs(output_folder +'/validation/labels/')
        dest = output_folder +'/validation/labels/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in train_data:
        train_data_img.append(i.replace('labels', 'images')) 
    for i in test_data:
        test_data_img.append(i.replace('labels', 'images')) 
    for i in val_data:
        val_data_img.append(i.replace('labels', 'images')) 

    for i in train_data_img:
        if not os.path.exists(output_folder +'/training/images/'):
            os.makedirs(output_folder +'/training/images/')
        dest = output_folder +'/training/images/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in test_data_img:
        if not os.path.exists(output_folder +'/evaluation/images/'):
            os.makedirs(output_folder +'/evaluation/images/')
        dest = output_folder +'/evaluation/images/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    for i in val_data_img:
        if not os.path.exists(output_folder +'/validation/images/'):
            os.makedirs(output_folder +'/validation/images/')
        dest = output_folder +'/validation/images/' + i[-20:]
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.copy(i, dest)

    print("Successfully split dataset according to train-test-val")