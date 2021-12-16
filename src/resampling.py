import os
import numpy as np
import SimpleITK as sitk

# Local imports
import dataloader as dl


def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    """
    Resamples an image given some axis scale ratios.

    Parameters:
        itk_image (SimpleITK.SimpleITK.Image): A SimpleITK image.
        out_spacing (list<float>): Scale ratios for each axis.
        is_label (bool): If True, use nearest neighbour interpolation,
            otherwise use spline interpolation.
    
    Returns:
        (SimpleITK.SimpleITK.Image): The resampled image.
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image) 


def pad_centered(arr, shape):
    """
    Pads an array to a given shape, keeping the original array centered.

    Parameters:
        arr (numpy.ndarray): Array to pad
        shape (tuple<int>): Shape to which to pad.
    
    Returns:
        (numpy.ndarray): A center padded image.
    """
    assert len(arr.shape) == len(shape)
    assert [arr.shape[i] <= shape[i] for i in range(len(shape))]

    pad_sizes = [(shape[i] - arr.shape[i])//2 for i in range(len(shape))]
    
    padding = []
    for i, p in enumerate(pad_sizes):
        if (shape[i] - arr.shape[i]) % 2 == 0:
            padding.append((p, p))
        else:
            padding.append((p, p + 1))
    
    centered = np.pad(arr, padding, mode='constant')
    return centered


def resample_to_shape(arr, shape):
    """
    Resample an image to a given shape without any stretching/rescaling.

    Parameters:
        arr (numpy.ndarray): Array to resample.
        shape (tuple<int>): Shape to which to resample.
    
    Returns:
        (numpy.ndarray): The resampled image.
    """

    # Determine maximum possible scale
    scales = np.array(arr.shape)/np.array(shape)
    scale = scales.max()
    scales = [scale]*len(arr.shape)

    # Resample the image to the maximum possible scale
    image = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    resampled_image = resample_image(image, out_spacing=scales)
    resampled_arr = sitk.GetArrayFromImage(resampled_image)
    
    # Pad to the correct shape
    resampled_arr = pad_centered(resampled_arr, shape)
    return resampled_arr

def main():
    atlas = dl.load_file("D:/CourseWork/CSC494/BrainTissueRegistration/data/MNI152_T1_0.7mm_brain_256resampled_norm.nii.gz")
    print(atlas.shape)

    shape = (64, 64, 64)
    
    if os.name == "nt":
        resampled = resample_to_shape(atlas, shape)
    else:
        import ants
        resampled = ants.resample_image(atlas, shape, 1, 0)
    
    print(resampled.shape)


if __name__ == "__main__":
    main()