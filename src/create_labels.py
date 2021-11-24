import numpy as np
import os

import dataloader as dl

def main():
    atlas_file = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/MNI152_T1_0.7mm_brain.nii.gz"
    data_dir = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/HCP_351_T1w_restore_brain"
    out_dir = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/HCP_351_T1w_restore_brain_labels"

    atlas = dl.load_file(atlas_file)
    shape = atlas.shape
    labels = np.ones(shape)

    print("atlas.shape", atlas.shape)
    
    for i, file in enumerate(os.listdir(data_dir)):
        if i == 0:
            print("data.shape", dl.load_file(os.path.join(data_dir, file)).shape)
        
        dl.save_nii_file(os.path.join(out_dir, file), labels)


        
if __name__ == "__main__":
    main()
