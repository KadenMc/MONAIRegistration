import os
import argparse
import matplotlib.pyplot as plt

import argparsing as ap
import dataloader as dl
import preprocessing as pre

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Path argument
    parser.add_argument('data', type=ap.file_or_dir_path, help='Path to data file or folder')
    parser.add_argument('save_path', type=ap.save_file_or_dir_path, help='Path to save data file or folder')
    parser.add_argument('-t', '--thresh', type=float, help='Binarize threshold. Defaults to individual data mean')
    parser.add_argument("--check", action="store_true", help="If flagged, visualizes images and generated labels")
    args = parser.parse_args()
    return args


def binarize(arr, args):
    if args.check:
        from visualize import visualize_3d
        visualize_3d(arr)
        print("Mean:", arr.mean())
    
    thresh = arr.mean() if args.thresh is None else args.thresh
    arr[arr < thresh] = 0
    arr[arr > thresh] = 1
    
    if args.check:
        from visualize import visualize_3d
        visualize_3d(arr)
    
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