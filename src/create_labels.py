import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import argparsing as ap
import preprocessing as pre

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Path argument
    parser.add_argument('data', type=ap.path, help='Path to data file or folder')
    parser.add_argument('save_path', type=ap.save_path, help='Path to save data file or folder')
    parser.add_argument('-t', '--thresh', type=float, help='Binarize threshold. Defaults to individual data mean')
    parser.add_argument("--fill", action="store_true", help="If flagged, fill gaps in the volumetric data")
    parser.add_argument("--visualize", action="store_true", help="If flagged, visualizes images and generated labels")
    args = parser.parse_args()
    return args


def binarize(arr, args):
    """
    Threshold a 3D image to create a binarized region of interest label.

    Parameters:
        arr (numpy.ndarray): 3D image.
        args (argparse.Namespace):
            <thresh> (float): Binarize threshold.
            <fill> (bool): Whether to fill enclosed gaps in the volume.
            <visualize> (bool): Whether to visualize the image and label.
    
    Returns:
        (numpy.ndarray): Binarized region of interest label.
    """
    if args.visualize:
        from visualize import visualize_3d
        visualize_3d(arr)
        print("Mean:", arr.mean())
    
    thresh = arr.mean() if args.thresh is None else args.thresh
    arr[arr < thresh] = 0
    arr[arr > thresh] = 1
    
    if args.fill:
        import fill_voids
        arr = fill_voids.fill(arr, in_place=True)
        
    if args.visualize:
        from visualize import visualize_3d, visualize_binary_3d
        visualize_3d(arr)
        visualize_binary_3d(arr)
        
    return arr


def main():
    args = parse_arguments()
    
    # Preprocess a single file
    if os.path.isfile(args.data):
        assert os.path.isfile(args.data)
        pre.preprocess_file(args.data, save_path=args.save_path, process_fn=binarize, process_args=args)
    
    # Preprocess a directory
    elif os.path.isdir(args.data):
        assert os.path.isdir(args.data)
        assert os.path.isdir(args.save_path)
        pre.preprocess_dir(args.data, save_path=args.save_path, process_fn=binarize, process_args=args)
    
    else:
        raise Exception("{} should be an existing file or directory.".format(args.data))


if __name__ == "__main__":
    main()