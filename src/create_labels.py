import os
import numpy as np
import matplotlib.pyplot as pl

import dataloader as dl
from visualize import visualize_3d


def binarize(arr, thresh=None):
    if thresh is None:
        thresh = arr.mean()
    
    arr[arr < thresh] = 0
    arr[arr > thresh] = 1
    return arr


def main(atlas_file, atlas_out, data_dir, out_dir):
    atlas = dl.load_file(atlas_file)
    #plt.hist(atlas.flatten(), bins=50)
    #plt.show()
    #visualize_3d(atlas)
    atlas = binarize(atlas, thresh=atlas.mean()/2)
    #visualize_3d(atlas)
    dl.save_nii_file(atlas_out, atlas)

    for i, file in enumerate(os.listdir(data_dir)):
        arr = dl.load_file(os.path.join(data_dir, file))
        #plt.hist(arr.flatten(), bins=50)
        #plt.show()
        #visualize_3d(arr)
        arr = binarize(arr)
        #visualize_3d(arr)
        dl.save_nii_file(os.path.join(out_dir, file), arr)


def check(out_dir):
    for i, file in enumerate(os.listdir(out_dir)):
        arr = dl.load_file(os.path.join(out_dir, file))
        visualize_3d(arr)


        
if __name__ == "__main__":
    atlas_file = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/MNI152_T1_0.7mm_brain.nii.gz"
    atlas_out = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/MNI152_T1_0.7mm_brain_labels.nii.gz"
    data_dir = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/HCP_351_T1w_restore_brain"
    out_dir = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/HCP_351_T1w_restore_brain_labels"
    #main(atlas_file, atlas_out, data_dir, out_dir)
    check(out_dir)
