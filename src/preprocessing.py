import numpy as np
import os
import argparse

import argparsing as ap
import dataloader as dl
import visualize as vis

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Path argument
    parser.add_argument('data', type=ap.path, help='Path to data file or directory')
    parser.add_argument('-s', '--save_path', type=ap.save_path, help='Path to save data file or directory')
    parser.add_argument('--slices', action='store_true', \
        help="If flagged, data is expected to be a directory of image slices")
    parser.add_argument('--slices_dir', action='store_true', \
        help="If flagged, data is expected to be a directory of directories of image slices")
    
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
        help="Resample to shape (a string of comma-delimited integers), e.g., '(256, 256, 256)'")
    parser.add_argument("--resample_scales", type=ap.delimited_floats, default=None, \
        help="Resample by axis scale (a string of comma-delimited floats), e.g., '(1.2, 0.9, 1)'")
    parser.add_argument("--normalize", action='store_true', \
        help="If flagged, min-max normalize the data")
    #slices: False
    #reorder_axes: [0, 2, 1]
    #rotate: [[1, [1, 2]], [2, [0, 2]]]
    #pad_shape
    #normalize
    
    args = parser.parse_args()
    return args


def rotate90(arr, k, axes=(0, 1)):
    """
    Rotate an array by 90 degrees k times in the plane specified by axes.

    Parameters:
        arr (numpy.ndarray): Array with two or more dimensions.
        k (int): Rotate the array 90 degrees k number of times
        axis (tuple<int>): Axes around which to rotate
    
    Returns:
        (numpy.ndarray): Rotated array.
    """
    return np.rot90(arr, k, axes)


def reorder_axes(arr, axes):
    """
    Reorder array axes.

    Parameters:
        arr (numpy.ndarray): An array.
        axes (list<ints>, tuple<ints>): The new axes order.
    
    Returns:
        (numpy.ndarray): Array with reordered axes.
    """
    return np.moveaxis(arr, [i for i in range(len(axes))], axes)


def min_max_normalize(arr):
    """
    Perform min-max normalization on the data.

    Parameters:
        arr (numpy.ndarray): An array.
    
    Returns:
        (numpy.ndarray): Min-max normalized array.
    """
    return arr - arr.min()/(arr.max() - arr.min())


def affine_transform(arr, theta=0, trans=[0, 0, 0], scale=[1, 1, 1], \
    output_shape=None):
    """
    Performs an affine transformation on a 3D image given rotation, translation, and scaling.

    Parameters:
        arr (numpy.ndarray): 3D image.
        theta (float): Rotation in radians.
        trans (list<float>): Translation in each axis.
        scale (list<float>): Scaling in each axis.
        output_shape (tuple<int>): Output image shape.
    
    Returns:
        transformed (numpy.ndarray): The transformed image.
    """
    # Compose the corresponding affine transformation matrix
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    trans_matrix = np.array([
        [1, 0, 0, trans[0]],
        [0, 1, 0, trans[1]],
        [0, 0, 1, trans[2]],
        [0, 0, 0, 1]
    ])
    
    scale_matrix = np.array([
        [scale[0], 0, 0, 0],
        [0, scale[1], 0, 0],
        [0, 0, scale[2], 0],
        [0, 0, 0, 1]
    ])
    
    matrix = np.matmul(scale_matrix, trans_matrix)
    matrix = np.matmul(matrix, rot_matrix)
    
    # Transform the image
    if output_shape is None:
        output_shape = arr.shape
    else:
        assert len(output_shape) == 3
    
    from scipy.ndimage import affine_transform
    transformed = affine_transform(arr, matrix, output_shape=output_shape)
    
    return transformed


def preprocess(arr, args):
    """
    A preprocessing pipeline for images.

    Parameters:
        arr (numpy.ndarray): An array.
        args (argparse.Namespace):
            resample_shape (list<float>, tuple<float>, None): Shape to which
                to resample.
            resample_scales (list<float>, tuple<float>, None): Axis scales to
                which to resample.
            normalize (bool): Whether to min-max normalize the image.
            visualize (bool): Whether to visualize the image before and after.
    """
    
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
                arr = rotate90(arr, lst[0], tuple(lst[1]))
        # Otherwise perform one rotation
        else:
            arr = rotate90(arr, config['rotate'][0], tuple(config['rotate'][1]))
    
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
    process_fn=preprocess, process_args=None, save_fn=dl.save_nii_file, \
    parallel=True, processes=1):
    """
    Preprocess slice files. Assumes that slices are ordered by name.

    Parameters:
        data (str): Slices directory path.
        save_path (str, None): Path to which to save the output. Will not save
            if None.
        load_fn (func): Function to load the image slices.
        process_fn (func): Function to preprocess the data.
        process_args (dict, argparse.Namespace): Arguments passed into the
            preprocessing function.
        save_fn (func): Function to save the data.
        parallel (bool): Whether to load the slices in parallel, or sequentially.
        processes (int): Number of processes to use during parallel loading.

    Returns:
        (numpy.ndarray): Preprocessed image created from slices.
    """
    # Load
    arr = load_fn(data, parallel=parallel, processes=processes)
    
    # Preprocess
    arr = process_fn(arr, process_args)
    
    # Save
    if save_path is not None:
        save_fn(save_path, arr)
    
    return arr


def preprocess_slices_dir(data, save_path=None, load_fn=dl.load_3d, process_fn=preprocess, \
    process_args=None, save_fn=dl.save_nii_file, parallel=True, processes=1):
    """
    Preprocess a directory of slice files.

    Parameters:
        data (str): Directory path full of slice directories.
        save_path (str, None): Path to which to save the output. Will not save
            if None.
        load_fn (func): Function to load the image slices.
        process_fn (func): Function to preprocess the data.
        process_args (dict, argparse.Namespace): Arguments passed into the
            preprocessing function.
        save_fn (func): Function to save the data.
        parallel (bool): Whether to load the slices in parallel, or sequentially.
        processes (int): Number of processes to use during parallel loading.

    Returns:
        (numpy.ndarray): Preprocessed image created from slices.
    """
    for d in os.listdir(data):
        save = None if save_path is None else ap.join(save_path, d)
        preprocess_slices(ap.join(data, d), save_path=save, load_fn=load_fn, process_fn=process_fn, \
            process_args=process_args, save_fn=save_fn, parallel=parallel, processes=processes)


def preprocess_file(file, save_path=None, load_fn=dl.load_file, \
    process_fn=preprocess, process_args=None, save_fn=dl.save_nii_file):
    """
    Preprocess an image loaded from file.

    Parameters:
        file (str): Path to image file.
        save_path (str, None): Path to which to save the output. Will not save
            if None.
        load_fn (func): Function to load the image slices.
        process_fn (func): Function to preprocess the data.
        process_args (dict, argparse.Namespace): Arguments passed into the
            preprocessing function.
        save_fn (func): Function to save the data.

    Returns:
        (numpy.ndarray): Preprocessed image.
    """
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
    """
    Preprocess a directory of images.

    Parameters:
        file (str): Path to directory.
        save_path (str, None): Path to which to save the output. Will not save
            if None.
        load_fn (func): Function to load the image slices.
        process_fn (func): Function to preprocess the data.
        process_args (dict, argparse.Namespace): Arguments passed into the
            preprocessing function.
        save_fn (func): Function to save the data.

    Returns:
        (numpy.ndarray): Preprocessed image.
    """
    for f in os.listdir(data):
        save = None if save_path is None else ap.join(save_path, f)
        preprocess_file(ap.join(data, f), save_path=save, load_fn=load_fn, process_fn=process_fn, \
            process_args=process_args, save_fn=save_fn)


def main():
    # Parse arguments
    args = parse_arguments()

    # Preprocess a directory of image slices (for a single image)
    if args.slices:
        assert os.path.isdir(args.data)
        preprocess_slices(args.data, save_path=args.save_path, process_args=args, \
            parallel=args.parallel, processes=args.processes)

    # Preprocess a directory of directories of image slices (for multiple images)
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