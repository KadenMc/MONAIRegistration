import numpy as np
import os
from os.path import join
import argparse

import argparsing as ap
import dataloader as dl
import visualize as vis

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Path argument
    parser.add_argument('data', type=ap.file_or_dir_path, help='Path to data file or folder')
    parser.add_argument('-s', '--save_path', type=ap.save_file_or_dir_path, help='Path to save data file or folder')
    parser.add_argument('--slices', action='store_true', \
        help="If flagged, data_path is expected to be a directory of image slices")
    parser.add_argument('--slices_dir', action='store_true', \
        help="If flagged, data_path is expected to be a directory of directories of image slices")
    
    # Multiprocessing arguments
    parser.add_argument("--sequential", action='store_true', \
        help="If flagged, do not load the data in parallel")
    parser.add_argument("--processes", type=int, default=4, \
        help="The number of processes for loading data in parallel")

    # Visualization arguments
    parser.add_argument("--visualize", action='store_true', \
        help="If flagged, the data before and after preprocessing")

    # Preprocessing arguments
    parser.add_argument("--resample_shape", type=ap.delimited_ints, default=None, \
        help="Resample to shape, e.g., '(256, 256, 256)'")
    parser.add_argument("--resample_scales", type=ap.delimited_floats, default=None, \
        help="Resample by axis scale, e.g., '(1.2, 0.9, 1)'")
    parser.add_argument("--normalize", action='store_true', \
        help="If flagged, min-max normalize the data")
    #slices: False
    #reorder_axes: [0, 2, 1]
    #rotate: [[1, [1, 2]], [2, [0, 2]]]
    #pad_shape
    #normalize
    
    args = parser.parse_args()
    return args


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
    
    
    """
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
    """

    assert len(arr.shape) == 3

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


    """
    return arr


def preprocess_slices(data, save_path=None, load_fn=dl.load_3d, \
    process_fn=preprocess, process_args=None, save_fn=dl.save_nii_file, parallel=True, processes=1):

    # Load
    arr = load_fn(data, True, parallel=parallel, processes=processes)
    
    # Preprocess
    arr = process_fn(arr, process_args)
    
    # Save
    if save_path is not None:
        save_fn(save_path, arr)
    
    return arr


def preprocess_slices_dir(data, save_path=None, load_fn=dl.load_3d, process_fn=preprocess, \
    process_args=None, save_fn=dl.save_nii_file, parallel=True, processes=1):
    for d in os.listdir(data):
        save = None if save_path is None else join(save_path, d)
        preprocess_slices(join(data, d), save_path=save, load_fn=load_fn, process_fn=process_fn, \
            process_args=process_args, save_fn=save_fn, parallel=parallel, processes=processes)


def preprocess_file(file, save_path=None, load_fn=dl.load_file, \
    process_fn=preprocess, process_args=None, save_fn=dl.save_nii_file):
    # Load
    arr = load_fn(file)
    
    # Process
    arr = process_fn(arr, process_args)
    
    # Save
    if save_path is not None:
        save_fn(save_path, arr)
    
    return arr


def preprocess_dir(data, save_path=None, load_fn=dl.load_file, \
    process_fn=preprocess, process_args=None, save_fn=dl.save_nii_file):
    for f in os.listdir(data):
        save = None if save_path is None else join(save_path, f)
        preprocess_file(join(data, f), save_path=save, load_fn=load_fn, process_fn=process_fn, \
            process_args=process_args, save_fn=save_fn)


def main():
    # Parse arguments
    args = parse_arguments()

    # Preprocess a folder of image slices (for a single image)
    if args.slices:
        assert os.path.isdir(args.data)
        preprocess_slices(args.data, save_path=args.save_path, process_args=args, \
            parallel=args.parallel, processes=args.processes)

    # Preprocess a folder of folders of image slices (for multiple images)
    elif args.slices_dir:
        assert os.path.isdir(args.data)
        preprocess_slices_dir(args.data, save_path=args.save_path, process_args=args, \
            parallel=args.parallel, processes=args.processes)
        
    # Preprocess a single file
    elif os.path.isfile(args.data):
        assert os.path.isfile(args.data)
        preprocess_file(args.data, save_path=args.save_path, process_args=args)
    
    # Preprocess a directory
    elif os.path.isdir(args.data):
        assert os.path.isdir(args.data)
        preprocess_dir(args.data, save_path=args.save_path, process_args=args)
    
    else:
        raise Exception("{} should be an existing file or directory.".format(args.data))


if __name__ == "__main__":
    main()