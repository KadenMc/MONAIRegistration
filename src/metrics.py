import numpy as np
from torch import Tensor

# Dice
from scipy.spatial.distance import dice
from monai.metrics.meandice import compute_meandice

# Cross-Correlation
from scipy.signal import correlate



def dice_score(y_pred, y):
    assert y_pred.shape == y.shape
    return 1 - dice(y_pred.flatten(), y.flatten())


def dice_score2(y_pred, y, include_background=True):
    return compute_meandice(y_pred, y, include_background=include_background)

def cross_correlation(y_pred, y):
    return correlate(y_pred, y, mode='same', method='auto')

if __name__ == "__main__":
    import dataloader as dl
    import visualize as vis
    import os
    from os.path import join
    
    y = dl.load_file("D:/CourseWork/CSC494/BrainTissueRegistration/data/MNI152_T1_0.7mm_brain_64resampled_norm.nii.gz")
    y_tensor = Tensor(y).unsqueeze(0).unsqueeze(0)
    print("y.shape", y.shape)
    
    #vis.visualize_3d(y)
    
    print("dice_score", dice_score(y, y))
    
    print("dice_score2", dice_score2(y_tensor, y_tensor, include_background=False).item())
    
    print("cross_correlation", cross_correlation(y, y))
    
    """
    print("\n")
    path = "D:/CourseWork/CSC494/BrainTissueRegistration/data/processed"
    for file in os.listdir(path):
        print(file)
        y_pred = dl.load_file(join(path, file))
        print("y_pred.shape", y_pred.shape)
    
        dice1 = dice_score(y_pred, y)
        print("dice_score", dice1)
    
        y_pred_tensor = Tensor(y_pred).unsqueeze(0).unsqueeze(0)
        dice2 = dice_score2(y_pred_tensor, y_tensor, include_background=False).item()
        print("dice_score2", dice2)
        print(dice1 - dice2)
        print("\n")
    """