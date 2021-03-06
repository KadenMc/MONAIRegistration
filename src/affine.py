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

# Local imports
import argparsing as ap
import preprocessing as pre
import visualize as vis


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Path argument
    parser.add_argument('moving', type=ap.path, help='Moving image file or directory')
    parser.add_argument('fixed', type=ap.file_path, help='Fixed image file')
    parser.add_argument('save_path', type=ap.path, help='Path to save data file or directory')
    parser.add_argument("--visualize", action="store_true", help="If flagged, visualizes images and generated labels")
    args = parser.parse_args()
    return args


def align_com(moving, static):
    """
    Align the center of mass of a moving image to that of a static image.

    Parameters:
        moving (numpy.ndarray): 3D moving image.
        static (numpy.ndarray): 3D static image.
    
    Returns:
        (numpy.ndarray): The center of mass aligned moving image.
    """
    from scipy.ndimage.measurements import center_of_mass
    moving_com = center_of_mass(moving)
    static_com = center_of_mass(static)
    delta = np.subtract(static_com, moving_com).astype(np.int64)
    moving = np.roll(np.roll(np.roll(moving, delta[0], axis=0), delta[1], axis=1), delta[2], axis=2)
    return moving


def register_affine(moving, args):
    """
    Affinely register a moving image to a fixed image.

    Parameters:
        moving (numpy.ndarray): 3D moving image.
        args (argparse.Namespace):
            <fixed> (numpy.ndarray): 3D fixed image.
            <visualize> (bool): Whether to visualize the input and transformed image.
    
    Returns:
        (numpy.ndarray): The affinely aligned moving image.
    """
    assert args.fixed.shape == moving.shape

    if args.visualize:
        vis.plot_slice_overlay(moving, args.fixed)

    # Begin with a simple center of mass alignment
    moving = align_com(moving, args.fixed)
    static = args.fixed

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
    
    if args.visualize:
        vis.plot_slice_overlay(transformed, args.fixed)
    
    # Return transformed image
    return transformed


def main():
    args = parse_arguments()
    
    # Load the fixed image and set it as one of the function arguments
    from dataloader import load_file
    args.fixed = load_file(args.fixed)

    # Preprocess a single file
    if os.path.isfile(args.moving):
        assert os.path.isfile(args.moving)
        pre.preprocess_file(args.moving, save_path=args.save_path, process_fn=register_affine, process_args=args)
    
    # Preprocess a directory
    elif os.path.isdir(args.moving):
        assert os.path.isdir(args.moving)
        assert os.path.isdir(args.save_path)
        pre.preprocess_dir(args.moving, save_path=args.save_path, process_fn=register_affine, process_args=args)
    
    else:
        raise Exception("{} should be an existing file or directory.".format(args.moving))


if __name__ == "__main__":
    main()