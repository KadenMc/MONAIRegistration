import numpy as np
import SimpleITK as sitk


def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
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


def pad_centered(a, shape):
    assert(len(a.shape) == len(shape))
    pad_sizes = [(shape[i] - a.shape[i])//2 for i in range(len(shape))]
    
    padding = []
    for i, p in enumerate(pad_sizes):
        if (shape[i] - a.shape[i]) % 2 == 0:
            padding.append((p, p))
        else:
            padding.append((p, p + 1))
    
    centered = np.pad(a, padding, mode='constant')
    return centered


def resample_to_shape(arr, shape):
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
    import dataloader as dl
    import visualize as vis
    
    import os
    
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