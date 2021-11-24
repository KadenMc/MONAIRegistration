import os
import argparse
import re

# Define a custom join function which replaces "\\" with "/",
# which can occasionally cause problems with filepaths otherwise
def join(a, b):
    return os.path.join(a, b).replace("\\", "/")

# Define relative paths
SRC_PATH = str(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
ROOT_PATH = SRC_PATH[:SRC_PATH.rindex('/')]
VISUALIZE_PATH = join(ROOT_PATH, 'visualizations')
MODEL_PATH = join(ROOT_PATH, 'models')
DATA_PATH = join(ROOT_PATH, 'data')


# Define argument parsing types
def path(path):
    if os.path.isdir(path) or os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid path".format(path))


def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid file path".format(path))


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid directory path".format(path))

def file_or_dir_path(path):
    if os.path.isdir(path) or os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid file or directory path".format(path))

def delimited_ints(s):
    if not isinstance(s, str):
        raise argparse.ArgumentTypeError("{} must be a string".format(s))
    
    # Keep only numbers, commas, and periods
    s = re.sub("[^0-9.,]", "", s)
    
    # Split and convert to integers
    try:
        return [int(i) for i in s.split(",")]
    except:
        raise argparse.ArgumentTypeError("Cannot convert {} to a list of integers".format(s))

def delimited_floats(s):
    if not isinstance(s, str):
        raise argparse.ArgumentTypeError("{} must be a string".format(s))
    
    # Keep only numbers, commas, and periods
    s = re.sub("[^0-9.,]", "", s)
    
    # Split and convert to floats
    try:
        return [float(i) for i in s.split(",")]
    except:
        raise argparse.ArgumentTypeError("Cannot convert {} to a list of floats".format(s))

def percent(p):
    try:
        p = float(p)
    except:
        return argparse.ArgumentTypeError("{} is not a valid percentage".format(p))
    
    if p < 0:
        return argparse.ArgumentTypeError("{} is not a valid percentage".format(p))
    # Check whether a decimal percentage was inputed
    if p > 1:
        if p > 100:
            return argparse.ArgumentTypeError("{} is not a valid percentage".format(p))
        return p/100
    return p


# Parse arguments
def parse_arguments_preprocessing():
    parser = argparse.ArgumentParser()

    # Path argument
    parser.add_argument('data', type=file_or_dir_path, help='Path to data file or folder')
    parser.add_argument('-s', '--save_path', type=str, help='Path to save data file or folder')
    parser.add_argument('--slices', action='store_true', \
        help="If flagged, data_path is expected to be a directory of image slices")
    parser.add_argument('--slices_dir', action='store_true', \
        help="If flagged, data_path is expected to be a directory of directories of image slices")
    
    # Multiprocessing arguments
    parser.add_argument("--sequential", action='store_true', \
        help="If flagged, do not load the data in parallel")
    parser.add_argument("--processes", type=int, default=4, \
        help="The number of processes for loading data in parallel")

    # Visualization arguments
    parser.add_argument("--visualize", action='store_true', \
        help="If flagged, the data before and after preprocessing")

    # Preprocessing arguments
    parser.add_argument("--resample_shape", type=delimited_ints, default=None, \
        help="Resample to shape, e.g., '(256, 256, 256)'")
    parser.add_argument("--resample_scales", type=delimited_floats, default=None, \
        help="Resample by axis scale, e.g., '(1.2, 0.9, 1)'")
    parser.add_argument("--normalize", action='store_true', \
        help="If flagged, min-max normalize the data")
    #slices: False
    #reorder_axes: [0, 2, 1]
    #rotate: [[1, [1, 2]], [2, [0, 2]]]
    #pad_shape
    #normalize
    
    args = parser.parse_args()
    return args


def parse_arguments_infer():
    parser = argparse.ArgumentParser()

    parser.add_argument('moving_file', type=file_path, help='Path to moving image')
    parser.add_argument('moving_labels_file', type=file_path, help='Path to moving image')
    parser.add_argument('fixed_file', type=file_path, help='Path to atlas file')
    parser.add_argument('fixed_labels_file', type=file_path, help='Path to fixed labels file')
    parser.add_argument('weights_file', type=file_path, help='Load model weights from file')
    
    args = parser.parse_args()
    return args


def parse_arguments_main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('images', type=file_or_dir_path, help='Path to folder of images')
    parser.add_argument('labels', type=file_or_dir_path, help='Path to folder of labels')
    parser.add_argument('atlas_file', type=file_path, help='Path to atlas file')
    parser.add_argument('atlas_label_file', type=file_path, help='Path to atlas file')
    parser.add_argument('--weights_file', type=file_path, help='Load model weights from file')
    parser.add_argument('--save_weights_file', default=join(MODEL_PATH, 'model.pth'), \
        help='Save model weights to file')
    parser.add_argument('--history', default=join(VISUALIZE_PATH, 'history.png'), \
        help='Path to save model history')
    
    # Training & data loading arguments
    parser.add_argument("--verbose", type=int, default=2, help="Training verbosity")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size - highly recommended to use the default, 1.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--val_interval", type=int, default=1, help="Calculatge validation every x epochs during training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--val_percent", type=percent, default=0.15, help="Validation dataset percentage")
    parser.add_argument("--test_percent", type=percent, default=0.15, help="Test dataset percentage")
    parser.add_argument("--num_workers", type=int, default=0, \
        help="Number of workers to perform multi-threading during caching. Default of 0 uses no multi-threading")
    parser.add_argument("--deterministic", action="store_true", help="If flagged, have deterministic training")

    args = parser.parse_args()
    return args