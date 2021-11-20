import numpy as np
import os
from os import listdir
from os.path import join, isdir, isfile


import argparsing as ap
import dataloader as dl
import visualize as vis

def rotate(arr, k, axes=(0, 1)):
    '''
    Rotate an array by 90 degrees k times in the plane specified by axes.

    arr: Array of two or more dimensions to rotate
    k: Rotate the array 90 degrees k number of times
    axis: Axes around which to rotate
    '''
    return np.rot90(arr, k, axes)


def reorder_axes(arr, axes):
    '''
    Re-order the array's axes.

    axes: The new axes order.
    '''
    return np.moveaxis(arr, [i for i in range(len(axes))], axes)


def min_max_normalize(arr):
    """
    Perform min-max normalization on the data.
    """
    return arr - arr.min()/(arr.max() - arr.min())

"""
def preprocess(arr, config, pad_shape=None):
    # Re-order axes
    if 'reorder_axes' in config:
        arr = reorder_axes(arr, config['reorder_axes'])

    # Rotate
    if 'rotate' in config:
        # Check whether there is multiple rotations
        if isinstance(config['rotate'][0], list):
            for lst in config['rotate']:
                arr = rotate(arr, lst[0], tuple(lst[1]))
        # Otherwise perform one rotation
        else:
            arr = rotate(arr, config['rotate'][0], tuple(config['rotate'][1]))
    
    # Pad to shape
    if pad_shape is not None:
        arr = pad_to_shape(arr, pad_shape)
    
    return arr
"""


def preprocess(arr, args):
    '''
    A preprocessing pipeline.
    '''
    
    # Visualize image before
    if args.visualize:
        vis.visualize_3d(arr)

    # Resample to a given shape
    if args.resample_shape is not None:
        assert args.resample_scales is None
        from resampling import resample_to_shape
        arr = resample_to_shape(arr, args.resample_shape)
    
    # Resample given axis scales
    if args.resample_scales is not None:
        assert args.resample_shape is None
        from resampling import resample_image
        arr = resample_image(arr, out_spacing=args.resample_scales)
    
    # Normalize
    if args.normalize:
        arr = min_max_normalize(arr)
    
    # Visualize image after
    if args.visualize:
        vis.visualize_3d(arr)
    
    return arr



def preprocess_slices(args, folder=None, file_out=None):
    # Define input path
    f = args.data if folder is None else folder
    assert isdir(f)
    
    # Define save path
    f_out = args.save_path if file_out is None else file_out
    
    # Load, preprocess, and save
    arr = dl.load_3d(f, True, parallel=args.parallel, processes=args.processes)
    arr = preprocess(arr, args)
    if f_out is not None:
        dl.save_nii_file(f_out, arr)
    
    return arr

def preprocess_slices_dir(args):
    if args.save_path is not None:
        assert isdir(args.save_path)
    
    # Load, preprocess, and save
    for d in listdir(args.data):
        save_path = join(args.save_path, d) if args.save_path is not None else None
        preprocess_slices(args, folder=join(args.data, d), file_out=save_path)
    

def preprocess_file(args, file=None, file_out=None):
    # Define input path
    f = args.data if file is None else file
    assert isfile(f)
    
    # Define save path
    f_out = args.save_path if file_out is None else file_out
    
    # Preprocess and save
    arr = preprocess(dl.load_file(f), args)
    if f_out is not None:
        dl.save_nii_file(f_out, arr)
    
    return arr

def preprocess_dir(args):
    if args.save_path is not None:
        assert isdir(args.save_path)
    
    # Preprocess and save
    for f in listdir(args.data):
        # If using extension .nii.gz, save as .nii
        from dataloader import check_extensions
        ext, _ = check_extensions(f)
        if ext == ".nii.gz":
            save_path = join(args.save_path, f[:-3]) if args.save_path is not None else None
        else:
            save_path = join(args.save_path, f) if args.save_path is not None else None
        
        preprocess_file(args, file=join(args.data, f), file_out=save_path)


def main():

    args = ap.parse_arguments_preprocessing()

    # Preprocess a folder of image slices (for a single image)
    if args.slices:
        assert isdir(args.data)
        preprocess_slices(args)

    # Preprocess a folder of folders of image slices (for multiple images)
    elif args.slices_dir:
        assert isdir(args.data)
        preprocess_slices_dir(args)
        
    # Preprocess a single file
    elif isfile(args.data):
        preprocess_file(args)
    
    # Preprocess a directory
    else:
        preprocess_dir(args)
        





    """
    if args.preprocess:
        # If flagged, preprocess and save data
        preprocess(config, parallel=(not args.sequential), processes=args.processes)
        return
    """

    #data = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/HCP_351_T1w_restore_brain_256resampled"
    #data_out = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/HCP_351_T1w_restore_brain_256resampled_norm"
    #normalize(data, data_out)

    #atlas = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/MNI152_T1_0.7mm_brain_256resampled.nii.gz"
    #atlas_out = "/home/mckeenka/projects/rrg-mgoubran/deepreg/data/MNI152_T1_0.7mm_brain_256resampled_norm.nii.gz"
    #normalize(atlas, atlas_out)
    """

    # Load file
    arr = dl.load_3d(args.file, args.slices, parallel=(not args.sequential), processes=args.processes)
    print("BEFORE - arr.shape", arr.shape)

    assert len(arr.shape) == 3
    
    # Visualize
    vis.visualize_3d(arr)

    # Get feedback on axes re-ordering and rotation
    print("Axes Re-order - e.g., '[0, 2, 1]' would switch the last two axes")
    axes = input("")
    axes = [int(i) for i in axes.strip('[] ').split(',')]

    print(axes)
    
    print("Rotations - Rotate the axes (AFTER REORDERED)")
    print("For each axis, write the number of rotations")

    rotations = []
    for i in range(len(arr.shape)):
        rotation = int(input())
        if rotation != 0:
            ax = [0, 1, 2]
            ax.remove(i)
            rotations.append([rotation, ax])

    print(rotations)

    config = {
        "reorder_axes": axes,
        "rotate": rotations
    }

    arr = preprocess(arr, config)
    print("AFTER - arr.shape", arr.shape)
    
    # Visualize after
    #vis.visualize_3d(arr)

    # Save oriented
    #dl.save_nii_file(file, arr)"""




if __name__ == "__main__":
    main()