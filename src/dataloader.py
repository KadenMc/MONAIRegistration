import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# Loading functions
def load_numpy(file):
    """
    Loads a .npy or .npz file.

    Parameters:
        file (str): File to load.
    
    Returns:
        (numpy.ndarray): Loaded file.
    """
    return np.load(file)


def load_nii_file(file):
    """
    Loads a .nii or .nii.gz file.

    Parameters:
        file (str): File to load.
    
    Returns:
        (numpy.ndarray): Loaded file.
    """
    img = nib.load(file)
    return img.get_fdata()


def load_tif_file(tif_file):
    """
    Loads a .tif file.

    Parameters:
        file (str): File to load.
    
    Returns:
        (numpy.ndarray): Loaded file.
    """
    return numpy.array(plt.imread(tif_file))


def format_data(moving, fixed, moving_labels=None, fixed_label=None):
    """
    Formats the data in MONAI's dictionary formatting style.
    Does not include the labels if they are None.

    Parameters:
        moving (str): Path to moving image file or directory.
        fixed (str): Path to fixed image file.
        moving_labels (str, None): Path to moving label file or directory.
            Optionally None.
        fixed_label (str, None): Path to fixed label file or directory.
            Optionally None.
    
    Returns:
        data_dict (dict of str: int): Dictionary formatted data.
    """

    # If the moving image path is a file
    if os.path.isfile(moving):
        data_dicts = [{
            "fixed_image": fixed,
            "moving_image": moving,
        }]
        
        if fixed_label is not None:
            data_dicts[0]["fixed_label"] = fixed_label
        
        if moving_labels is not None:
            data_dicts[0]["moving_label"] = moving_labels
        
    # If the moving image path is a directory
    elif os.path.isdir(moving):
        # Get files in images and corresponding labels directories
        # Necessary to sort for the below assertion
        images = sorted(os.listdir(moving))

        if moving_labels is not None:
            labels = sorted(os.listdir(moving_labels))
            
            # Assert the images and labels directories have the same files
            assert len(images) == len(labels) and images == labels

        else:
            labels = None

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
    """
    Splits a set of files into training, validation, and testing sets.
    The validation and testing percentages are provided, where the training
    percent makes up the remainder of the dataset.

    Parameters:
        files (list, numpy.ndarray): Files to be split up.
        val_percent (float): Decimal percentage of files to have in validation set.
        test_percent (float): Decimal percentage of files to have in testing set.
        randomize (bool): Whether to randomize the files before splitting.
    
    Returns:
        train_files (list, numpy.ndarray): Files in the training set.
        val_files (list, numpy.ndarray): Files in the validation set.
        test_files (list, numpy.ndarray): Files in the testing set.
    """
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


# Saving functions
def save_numpy(file, arr):
    """
    Saves a .npy or .npz file.

    Parameters:
        file (str): Save file path.
        arr (np.ndarray): Array to save.
    """
    np.save(file, arr)


def save_nii_file(file, arr):
    """
    Saves a .nii or .nii.gz file.

    Parameters:
        file (str): Save file path.
        arr (np.ndarray): Array to save.
    """
    img = nib.Nifti1Image(arr, np.eye(4))
    img.to_filename(file)


def extension_matches(file, extension):
    """
    Check whether a file has an extension.

    Parameters:
        file (str): File path
        extension (str): Extension

    Returns:
        (bool): Whether the file has the extension.
    """
    # Add a beginning period if not already there
    if extension[0] != '.':
        extension = '.' + extension

    # Return whether the file has the extension
    if file[-len(extension):] == extension:
        return True
    return False


def get_recognized_extension(file):
    """
    Extracts and returns a recognized extension. If the extension is not
    recognized, then return None.

    Parameters:
        file (str): File from which to extract an extension.
    
    Returns:
        (str, None): Returns a recognized extension, or returns None.
    """
    for key in EXTS:
        if extension_matches(file, key):
            return key
    
    return None

def get_loading_function(file, ignore_unknown=False):
    """
    Gets the corresponding loading function for a recognized file extension/type.

    Parameters:
        file (str): File for which to find a loading function.
        ignore_unknown (bool): How to handle the case when the extension is not
            recognized, either raising an error, or returning None.
    
    Returns:
        (func, None): Returns a loading function for a recognized extension, or
            None if not recognized and ignore_unknown is True.
    """
    # Get extension
    ext_key = get_recognized_extension(file)

    # Check whether it is a recognized extension
    if ext_key is None:
        if ignore_unknown:
            return None
        else:
            raise Exception("File extension not supported")
    
    return EXTENSION_TO_FN_DICT[ext_key]


def load_file(file):
    """
    A general-purpose function to load a file of recognized type.
    Raises an error if the extension is not recognized.

    Parameter:
        file: File to load.
    
    Returns:
        (numpy.ndarray): Loaded file.
    """
    # Check extension and get the corresponding data loading function
    load_fn = get_loading_function(file)
    return load_fn(file)


def load_files_parallel(files, processes=os.cpu_count() - 1):
    """
    Loads data files in parallel.

    Parameters:
        files (list<str>): An ordered list of the files to load.
        processes (int): The number of processes.
    
    Returns:
        (list<numpy.ndarray>): A list of the loaded files.
    """
    from multiprocessing import Pool
    p = Pool(processes)
    return p.map(load_file, files)


def get_files(path, check_extension=False):
    """
    Returns files in a directory.

    Parameters:
        path (str): Directory path.
        check_extension (bool): Whether to check the file extension before yielding a file.
            If true, it will not yield a file with an unrecognized file extension.
    """
    assert os.path.isdir(path)
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if check_extension:
                if get_recognized_extension(file) is not None:
                    yield file
            else:
                yield file


def organize_data_files(data_paths):
    """
    Takes a list of files or directories, and returns only files with valid extensions.

    Parameters:
        data_paths (list<str>): A list of training files or directories.
    
    Returns:
        (list<str>): A list of training files or directories with valid extensions.
    """
    file_paths = []
    for d in data_paths:
        # If directory
        if os.path.isdir(d):
            file_paths.extend(list(get_files(d, check_extension=True)))
        # If a file
        else:
            if get_recognized_extension(d) is not None:
                file_paths.append(d)
    return file_paths


def load_3d_slices(files, parallel=True, processes=os.cpu_count() - 1):
    """
    Loads 3D slice files in parallel or sequentially, combining them into a
    slice image at the end. Assumes files are ordered in order of the slices.
    Assumes all slice files have the same shape.

    Parameters:
        files (list<str>): Ordered file paths.
        parallel (bool): Whether to load the files sequentially or in parallel.
        processes (int): Number of processes/workers during parallel loading.
    
    Returns:
        (numpy.ndarray): Combined image created from slice files.
    """
    # If parallel, load slices in parallel
    if parallel:
        x = np.dstack(load_files_parallel(files, processes=processes))
    # Otherwise, load slices sequentially
    else:
        x = np.dstack([load_file(f) for f in files])


def load_3d(path, parallel=True, processes=os.cpu_count() - 1):
    """
    This function loads a 3D image either from a file, or a directory of
    same-shape slice files. 
    

    Parameters:
        path (str): If slices is True, path must be a directory of files which,
            when ordered by name, represent the ordered slices of a 3d image.
            If slices is False, path must be an image file.
        
        parallel (bool): Whether to load the slice files sequentially or in parallel.
        processes (int): Number of processes during parallel loading of slices.

    Returns:
        (numpy.ndarray): Loaded image.
    """
    # Load the data from several slices
    if os.path.isdir(path):
        # Organize slice files in a given directory
        files = [os.path.join(path, f) for f in get_files(path)]
        files = np.array(files)
        files.sort()
        return load_3d_slices(files, parallel=parallel, processes=processes)

    # Otherwise, load the data from a single file
    return load_file(path)


def determine_resize_shape(fixed_file, resize_shape, resize_ratio, d=1):
    """
    Determine the resize shape given a resize shape and ratio.
    If resize_ratio is not None, determine the shape using this.
    Otherwise, if resize_shape is not None, determine the shape using this.
    Otherwise, determine the shape using the current shape.

    Determines resize shape as detailed above, and then finds the closest
    viable shape, there its axes are divisible by d.

    Parameters:
        fixed_file (str): Fixed image file path.
        resize_shape (tuple<int>, None): Resize shape.
        resize_ratio (float): Resize ratio.
        d (int): Integer by which the shape axes must be divisible.
            There is no effect if d is 1.
    """
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
    """
    Filters items out of lst if they aren't in keys.

    Parameters:
        keys (list): List of keys.
        lst (list): List to filter by keys.
    
    Returns:
        (list): The filtered list.
    """
    return [i for i in lst if i in keys]


def create_transforms(fixed_file, keys, augment_data=True, resize_shape=None, \
    resize_ratio=None, d=8):
    """
    Creates the MONAI transforms necessary for image registration.

    Parameters:
        fixed_file (str): Path to fixed image file - used to calculate resize shapes.
        keys (list<str>): Keys are some subset of "fixed_image", "moving_image",
            "fixed_label", and "moving_label" depending on the files provided.
        augment_data (bool): Whether to include data augmentation transforms.
            Recommended during training, but not during inference.
        resize_shape (tuple<int>, None): Resize shape.
        resize_ratio (float): Resize ratio.
        d (int): Integer by which the shape axes must be divisible.
            There is no effect if d is 1.

    Returns:
        (monai.transforms.compose.Compose) Composition of sequential transforms.
    """
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
    shape = determine_resize_shape(fixed_file, resize_shape, resize_ratio, d=d)
    
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
 

def create_train_dataloaders(args, train_files, val_files, visualize=False, visualize_path=None):
    """
    Creates training and validation dataloaders for training.

    Parameters:
        args (argparse.Namespace):
            <fixed> (numpy.ndarray): 3D fixed image file path.
            <resize_shape> (numpy.ndarray): Resize shape.
            <resize_ratio> (numpy.ndarray): Resize ratio.
            <cache_rate> (numpy.ndarray): Percent of training dataset to cache at once.
            <cache_num> (numpy.ndarray): Number of files from training dataset to cache at once.
            <num_workers> (numpy.ndarray): Number of workers for parallelization.
            <batch_size> (numpy.ndarray): Training batch size.
        train_files (list<dict>): List of training formatted image dictionaries.
        val_file (list<dict>): List of validation formatted image dictionaries.
        visualize (bool): Whether to visualize data from the training data.
        visualize_path (str, None): A path to save the training data visualization.
            Does not save if it is None, or visualize is False.
    
    Returns
        train_loader (monai.data.DataLoader): Training dataloader.
        val_loader (monai.data.DataLoader): Validation dataloader.
    """
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


def create_dataloader_infer(files, fixed, resize_shape=None, resize_ratio=None):
    """
    Creates dataloader for inference.

    Parameters:
        files (list<dict>): List of formatted image dictionaries.
        fixed (numpy.ndarray): 3D fixed image file path.
        resize_shape (numpy.ndarray): Resize shape.
        resize_ratio (numpy.ndarray): Resize ratio.
    
    Returns
        loader (monai.data.DataLoader): Inference dataloader.
    """
    from monai.data import DataLoader, CacheDataset
    
    # Get data transforms
    transforms = create_transforms(fixed, keys=list(files[0].keys()), \
        augment_data=False, resize_shape=resize_shape, resize_ratio=resize_ratio)
    
    ds = CacheDataset(data=files, transform=transforms, cache_rate=1.0)
    loader = DataLoader(ds, batch_size=1)
    return loader


# Extensions which can be loaded, and their corresponding functions
EXTENSION_TO_FN_DICT = {
    '.npy': load_numpy,
    '.npz': load_numpy,
    '.tif': load_tif_file,
    '.nii': load_nii_file,
    '.nii.gz': load_nii_file,
}
EXTS = list(EXTENSION_TO_FN_DICT.keys())
EXTS.sort(key=len, reverse=True)