import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def format_data(moving, fixed, moving_labels, fixed_label):
    if os.path.isfile(moving):
        data_dicts = [{
            "fixed_image": fixed,
            "moving_image": moving,
        }]
        
        if fixed_label is not None:
            data_dicts[0]["fixed_label"] = fixed_label
        
        if moving_labels is not None:
            data_dicts[0]["moving_label"] = moving_labels
        
    elif os.path.isdir(moving):
        # Get files in images and corresponding labels directories
        # Necessary to sort for the below assertion
        images = sorted(os.listdir(moving))

        if moving_labels is not None:
            labels = sorted(os.listdir(moving_labels))
        else:
            labels = None

        # Assert the images and labels folders have the same files
        assert len(images) == len(labels) and images == labels

        data_dicts = []
        for i, image in enumerate(images):
            data_dicts.append({
                    "fixed_image": fixed,
                    "moving_image": os.path.join(moving, image),
            })
            
            if fixed_label is not None:
                data_dicts[-1]["fixed_label"] = fixed_label
        
            if labels is not None:
                data_dicts[-1]["moving_label"] = os.path.join(moving_labels, labels[i])
    
    
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


def check_extensions(file, ignore_unknown=False, return_load_fn=False):
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
            if return_load_fn:
                return extension_dict[key]
            else:
                return key
    
    if not ignore_unknown:
        raise Exception("File extension not supported")


def load_file(file):
    '''
    A generalized function to load common files types in registration.

    file: The file to load.
    '''

    # Check extension and get the corresponding data loading function
    load_fn = check_extensions(file, return_load_fn=True)
    return load_fn(file)


def load_files_parallel(files, processes=os.cpu_count() - 1):
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
                if check_extensions(file, ignore_unknown=True) is not None:
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
            if check_extensions(d, ignore_unknown=True) is not None:
                file_paths.append(d)
    return file_paths


def load_3d_slices(files, parallel=True, processes=os.cpu_count() - 1):
    # If parallel, load slices in parallel
    if parallel:
        x = np.dstack(load_files_parallel(files, processes=processes))
    # Otherwise, load slices sequentially
    else:
        x = np.dstack([load_file(f) for f in files])


def load_3d(path, slices, parallel=True, processes=os.cpu_count() - 1):
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

def determine_resize_shape(fixed_file, resize_shape, resize_ratio, d):
    # Load the fixed image to get the shape info
    fixed = load_file(fixed_file)
    
    # Use a ratio
    if resize_ratio is not None:        
        # Use the ratio to calculate an approximate (non-integer) shape
        shape = [i*resize_ratio for i in fixed.shape]
        
    # Use a provided size
    elif resize_shape is not None:
        shape = resize_shape
    
    # Use the fixed image size
    else:
        shape = fixed.shape
    
    # In order to work with some architectures,
    # use the closest shape divisible by d
    if d > 1:
        shape = tuple([int(i - (i % d)) if i % d <= d/2 else \
            int(i + (d - (i % d))) for i in shape])
    else:
        shape = tuple([int(i) for i in shape])
    
    return shape

def keep_in_keys(keys, lst):
    return [i for i in lst if i in keys]


def create_transforms(fixed_file, keys, augment_data=True, resize_shape=None, resize_ratio=None, d=8):
    from monai.transforms import (
        AddChanneld,
        Compose,
        LoadImaged,
        RandAffined,
        Resized,
        ScaleIntensityd,
        EnsureTyped,
    )
    
    # Note: In the transforms, the hanging 'd' is necessary for the data
    # format using dictionaries
    transforms = [
        # Load the data
        LoadImaged(
            keys=keep_in_keys(keys, ["fixed_image", "moving_image", "fixed_label", "moving_label"])
        ),
        # Add a channel to the data
        AddChanneld(
                keys=keep_in_keys(keys, ["fixed_image", "moving_image", "fixed_label", "moving_label"])
        ),
        # Min-max normalize the intensity of the fixed and moving images
        ScaleIntensityd(
            keys=keep_in_keys(keys, ["fixed_image", "moving_image"])
        )
    ]
    
    # Determine shape to which we'll resize the data
    shape = determine_resize_shape(fixed_file, resize_shape, resize_ratio, d)
    
    # Augment data - used only during training
    if augment_data:
        """
        # Augment the data to twice the size of this shape
        augment_shape = tuple([i*2 for i in shape])
        
        transforms.append(
            # Perform a random affine transformation on the data
            RandAffined(
                keys=["fixed_image", "moving_image", "fixed_label", "moving_label"],
                mode=('bilinear', 'bilinear', 'nearest', 'nearest'),
                prob=1.0, spatial_size=augment_shape,
                rotate_range=(0, 0, np.pi / 15), scale_range=(0.1, 0.1, 0.1)
            )
        )
        """
        pass

    # Resize the data
    print("Resizing to shape {}".format(shape))
    resize_keys = keep_in_keys(keys, ["fixed_image", "moving_image", "fixed_label", "moving_label"])
    resize_mode = ["trilinear" if "image" in i else "nearest" for i in resize_keys]
    resize_align_corners = [True if "image" in i else None for i in resize_keys]
    transforms.extend([
        # Resize the image
        Resized(
            keys=resize_keys,
            mode=resize_mode,
            align_corners=resize_align_corners,
            spatial_size=shape
        )
    ])
    
    transforms.extend([
        # Ensure the input data to be a PyTorch Tensor or numpy array
        EnsureTyped(
            keys=keep_in_keys(keys, ["fixed_image", "moving_image", "fixed_label", "moving_label"])
        )
    ])
    
    return Compose(transforms)
    
    

def create_dataloaders(args, train_files, val_files, visualize=False, visualize_path=None):
    from monai.data import DataLoader, CacheDataset
    
    # Get data transforms
    train_transforms = create_transforms(args.fixed, keys=list(train_files[0].keys()), \
        augment_data=True, resize_shape=args.resize_shape, resize_ratio=args.resize_ratio)
    val_transforms = create_transforms(args.fixed, keys=list(val_files[0].keys()), \
        augment_data=False, resize_shape=args.resize_shape, resize_ratio=args.resize_ratio)
    
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

    # Create training dataset and dataloader
    train_ds = CacheDataset(data=train_files, transform=train_transforms,
        cache_rate=args.cache_rate, cache_num=args.cache_num, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Create validation dataset and dataloader
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    
    return train_loader, val_loader



def create_dataloader_infer(val_files, fixed, resize_shape=None, resize_ratio=None):
    from monai.data import DataLoader, CacheDataset
    
    # Get data transforms
    transforms = create_transforms(fixed, keys=list(val_files[0].keys()), \
        augment_data=False, resize_shape=resize_shape, resize_ratio=resize_ratio)
    
    ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=1)
    return loader