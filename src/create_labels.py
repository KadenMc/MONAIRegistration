import numpy as np
import os

import dataloader as dl

data_dir = "D:/CourseWork/CSC494/BrainTissueRegistration/data/files"

atlas_file = "D:/CourseWork/CSC494/BrainTissueRegistration/data/MNI152_T1_0.7mm_brain_256resampled_norm.nii.gz"
atlas = dl.load_file(atlas_file)
shape = atlas.shape
labels = np.zeros(shape)

for file in os.listdir(data_dir):
    ext, _ = dl.check_extensions(file)
    fname = file[:-len(ext)] + "_labels.nii.gz"
    dl.save_nii_file(os.path.join(data_dir, fname), labels)