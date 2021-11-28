import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def format_data(moving, moving_labels, fixed, fixed_label):
    assert os.path.isfile(fixed)
    assert os.path.isfile(fixed_label)
    
    if os.path.isfile(moving):
        assert os.path.isfile(moving_labels)
        data_dicts = [{
            "fixed_image": fixed,
            "moving_image": moving,
            "fixed_label": fixed_label,
            "moving_label": moving_labels,
        }]
        
    elif os.path.isdir(moving):
        assert os.path.isdir(moving_labels)
        
        # Get files in images and corresponding labels directories
        # Necessary to sort for the below assertion
        images = sorted(os.listdir(moving))
        labels = sorted(os.listdir(moving_labels))

        # Assert the images and labels folders have the same files
        assert len(images) == len(labels) and images == labels

        data_dicts = []
        for i, image in enumerate(images):
            data_dicts.append(
                {
                    "fixed_image": fixed,
                    "moving_image": os.path.join(moving, image),
                    "fixed_label": fixed_label,
                    "moving_label": os.path.join(moving_labels, labels[i]),
                }
            )
    
    else:
        raise Exception("images must be a file or directory")
    
        
    return data_dicts


def split_dataset(files, val_percent=0.15, test_percent=0.15, randomize=False):
    assert val_percent + test_percent <= 1

    if randomize:
        np.random.shuffle(files)

    # Calculate number of files in the test and validation datasets
    test_n = int(len(files) * test_percent)
    val_n = int(len(files) * val_percent)

    # Split up the dataset
    test_files = files[:test_n]
    val_files = files[test_n: test_n + val_n]
    train_files = files[test_n + val_n:]

    return train_files, val_files, test_files


def load_numpy(file):
    return np.load(file)


def save_numpy(file, arr):
    np.save(file, arr)


def load_nii_file(file):
    img = nib.load(file)
    return img.get_fdata()


def save_nii_file(file, arr):
    img = nib.Nifti1Image(arr, np.eye(4))
    img.to_filename(file)


def load_tif_file(tif_file):
    return plt.imread(tif_file)


def extension_matches(file, extension):
    if file[-len(extension):] == extension:
        return True
    return False


def check_extensions(file):
    # Extensions which can be loaded, and their corresponding functions
    extension_dict = {
        '.npy': load_numpy,
        '.npz': load_numpy,
        '.tif': load_tif_file,
        '.nii': load_nii_file,
        '.nii.gz': load_nii_file,
    }

    # See if any extensions match, from longest extension to smallest
    # This ensures that, e.g., a '.nii.gz' won't be confused for '.gz'
    exts = list(extension_dict.keys())
    exts.sort(key=len, reverse=True)
    for key in exts:
        if extension_matches(file, key):
            return key, extension_dict[key]
    return None


def load_file(file):
    '''
    A generalized function to load common files types in registration.

    file: The file to load.
    '''

    # Check for extension and its corresponding data loading function
    args = check_extensions(file)

    # If None is returned, raise an error since the extension is not supported
    if args is None:
        raise Exception('This file type is not currently supported.')

    ext, func = args
    return func(file)


def load_files_parallel(files, processes=4):
    '''
    Load data files in parallel.

    files: An ordered list of the files to load
    processes: The number of processes
    '''
    from multiprocessing import Pool
    p = Pool(processes)
    return p.map(load_file, files)


def get_files(path, check_extension=False):
    '''
    Returns files in a directory.
    '''
    assert os.path.isdir(path)
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if check_extension:
                if check_extensions(file) is not None:
                    yield file


def organize_data_files(data_paths):
    '''
    Takes a list of training files and directories, and returns only training
    files which have valid extensions.
    '''
    file_paths = []
    for d in data_paths:
        # If directory
        if os.path.isdir(d):
            file_paths.extend(list(get_files(d, check_extension=True)))
        else:
            if check_extensions(d) is not None:
                file_paths.append(d)
    return file_paths


def load_3d_slices(files, parallel=True, processes=4):
    # If parallel, load slices in parallel
    if parallel:
        x = np.dstack(load_files_parallel(files, processes=processes))
    # Otherwise, load slices sequentially
    else:
        x = np.dstack([load_file(f) for f in files])


def load_3d(path, slices, parallel=True, processes=4):
    '''
    load_3d loads three different kinds of inputs.
    If slices is True, path must be a directory with files which, when ordered by name, represent the ordered slices of a 3d image.
    If slices is False and path is a directory, the function expects a folder of same-shape images.
    If slices is False and path is a file, it will simply load this file.

    path: A directory or file as described above
    slices: Whether or not the given directory
    parallel: Relevant when loading slices or same-shape imagess
    processes: When loading in parallel, this is how many processes to use

    '''
    # Load the data from several slices
    if slices:
        assert os.path.isdir(path)

        # Organize slice files in a given directory
        files = [os.path.join(path, f) for f in get_files(path)]
        files = np.array(files)
        files.sort()

        return load_3d_slices(files, parallel=parallel, processes=processes)

    # Otherwise, check if a directory - if so, load all the same-shape files from this directory
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in get_files(path)]
        if parallel:
            return np.stack(load_files_parallel(files, processes=processes), axis=0)
        else:
            x = np.stack([load_file(f) for f in files], axis=0)

    # Otherwise, load the data from a single file
    return load_file(path)





def create_dataloaders(args, train_files, val_files, visualize=False, visualize_path=None):
    from monai.data import DataLoader, CacheDataset
    from monai.transforms import (
        AddChanneld,
        Compose,
        LoadImaged,
        RandAffined,
        Resized,
        ScaleIntensityRanged,
        EnsureTyped,
    )
    
    if args.resample_shape is not None:
        if any([i % 2 == 1 for i in args.resample_shape]):
            raise Exception("Spatial dimensions be even. Please use a resample shape with even dimensions")
    
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
            AddChanneld(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
            ScaleIntensityRanged(
                keys=["fixed_image", "moving_image"],
                a_min=-285, a_max=3770, b_min=0.0, b_max=1.0, clip=True,
            ),
            RandAffined(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=('bilinear', 'bilinear', 'nearest', 'nearest'),
                prob=1.0, spatial_size=(192, 192, 208),
                rotate_range=(0, 0, np.pi / 15), scale_range=(0.1, 0.1, 0.1)
            ),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=('trilinear', 'trilinear', 'nearest', 'nearest'),
                align_corners=(True, True, None, None),
                spatial_size=(96, 96, 104)
            ),
            EnsureTyped(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
            AddChanneld(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
            ScaleIntensityRanged(
                keys=["fixed_image", "moving_image"],
                a_min=-285, a_max=3770, b_min=0.0, b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=('trilinear', 'trilinear', 'nearest', 'nearest'),
                align_corners=(True, True, None, None),
                spatial_size=(96, 96, 104)
            ),
            EnsureTyped(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
        ]
    )
    
    # Visualize dataset
    if visualize:
        from visualize import check_dataset
        check_dataset(train_files, train_transforms, save_path=visualize_path)
    
    
    # Use CacheDataset to accelerate training and validation process,
    # it's 10x faster than the regular Dataset. To achieve best performance,
    # set cache_rate=1.0 to cache all the data, if memory is not enough, set
    # lower value. Users can also set cache_num instead of cache_rate, will
    # use the minimum value of the 2 settings. And set num_workers to enable
    # multi-threads during caching.

    train_ds = CacheDataset(data=train_files, transform=train_transforms,
                            cache_rate=args.cache_rate, cache_num=args.cache_num, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = CacheDataset(data=val_files, transform=val_transforms,
                        cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    
    return train_loader, val_loader



def create_dataloader_infer(args, val_files):
    from monai.data import DataLoader, CacheDataset
    from monai.transforms import (
        AddChanneld,
        Compose,
        LoadImaged,
        Resized,
        ScaleIntensityRanged,
        EnsureTyped,
    )
    
    transforms = Compose(
        [
            LoadImaged(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
            AddChanneld(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
            ScaleIntensityRanged(
                keys=["fixed_image", "moving_image"],
                a_min=-285, a_max=3770, b_min=0.0, b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=('trilinear', 'trilinear', 'nearest', 'nearest'),
                align_corners=(True, True, None, None),
                spatial_size=tuple(args.resample_shape)
            ),
            EnsureTyped(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"]
            ),
        ]
    )
    
    ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=1)
    return loader