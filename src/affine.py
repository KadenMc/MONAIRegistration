import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from dipy.viz import regtools
from dipy.align.imaffine import (
    transform_centers_of_mass,
    AffineMap,
    MutualInformationMetric,
    AffineRegistration
)
from dipy.align.transforms import (
    TranslationTransform3D,
    RigidTransform3D,
    AffineTransform3D
)

import argparsing as ap
import preprocessing as pre
import visualize as vis


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Path argument
    parser.add_argument('data', type=ap.file_or_dir_path, help='Path to data file or folder')
    parser.add_argument('atlas', type=ap.file_or_dir_path, help='Path to data file or folder')
    parser.add_argument('save_path', type=ap.save_file_or_dir_path, help='Path to save data file or folder')
    parser.add_argument("--check", action="store_true", help="If flagged, visualizes images and generated labels")
    args = parser.parse_args()
    return args


def affine_transform(arr, theta=0, trans=[0, 0, 0], scale=[1, 1, 1], \
    output_shape=None):
    from scipy.ndimage import affine_transform
    
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
    
    if output_shape is None:
        output_shape = arr.shape
    
    return affine_transform(arr, matrix, output_shape=output_shape)


def align_com(moving, static):
    """
    Align center of mass of a moving image to a static image
    """
    from scipy.ndimage.measurements import center_of_mass
    moving_com = center_of_mass(moving)
    static_com = center_of_mass(static)
    delta = np.subtract(static_com, moving_com).astype(np.int64)
    moving = np.roll(np.roll(np.roll(moving, delta[0], axis=0), delta[1], axis=1), delta[2], axis=2)
    return moving, static


def register_affine(moving, args):
    assert fixed.shape == moving.shape
    
    vis.plot_slice(moving)    

    if args.check:
        vis.plot_slice_overlay(moving, fixed)

    # Begin with a simple center of mass alignment
    moving, static = align_com(moving, fixed)

    # Perform affine registration
    affreg = AffineRegistration()

    # Translation
    identity = np.eye(4)
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = identity
    translation = affreg.optimize(static, moving, transform, params0, starting_affine=starting_affine)

    # Translation and rotation
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0, starting_affine=starting_affine)

    # Full affine transform
    transformed = translation.transform(moving)
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0, starting_affine=starting_affine)
    
    # Get the registered image
    transformed = affine.transform(moving)
    
    if args.check:
        vis.plot_slice_overlay(transformed, fixed)
    
    # Return transformed image
    return transformed


def main():
    args = parse_arguments()
    
    from dataloader import load_file
    global fixed
    fixed = load_file(args.atlas)

    # Preprocess a single file
    if os.path.isfile(args.data):
        assert os.path.isfile(args.data)
        pre.preprocess_file(args.data, save_path=args.save_path, process_fn=register_affine, process_args=args)
    
    # Preprocess a directory
    elif os.path.isdir(args.data):
        assert os.path.isdir(args.data)
        assert os.path.isdir(args.save_path)
        pre.preprocess_dir(args.data, save_path=args.save_path, process_fn=register_affine, process_args=args)
    
    else:
        raise Exception("{} should be an existing file or directory.".format(args.data))


if __name__ == "__main__":
    main()