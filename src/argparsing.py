import os
import argparse


def join(a, b):
    """
    A custom path join function which replaces "\\" with "/". Otherwise, os.path.join can
    occasionally cause problems with filepaths (especially on Windows).
    """
    return os.path.join(a, b).replace("\\", "/")

# Define relative paths
SRC_PATH = str(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
ROOT_PATH = os.path.dirname(SRC_PATH)
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


def save_file_path(path):
    if os.path.isdir(os.path.dirname(path)):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not a valid directory path to save a file".format(os.path.dirname(path)))


def save_dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not an existing directory path".format(path))


def save_file_or_dir_path(path):
    if os.path.isdir(os.path.dirname(path)):
        return path
    else:
        raise argparse.ArgumentTypeError("{} is not valid save path".format(path))


def delimited_num(s, num_type):
    if not isinstance(s, str):
        raise argparse.ArgumentTypeError("{} must be a string".format(s))
    
    # Keep only numbers, commas, and periods
    import re
    s = re.sub("[^0-9.,]", "", s)
    
    # Split and convert to integers
    try:
        lst = [num_type(i) for i in s.split(",")]
    except:
        raise argparse.ArgumentTypeError("Cannot convert {} to a list of integers".format(s))

    # Check whether empty
    if len(lst) == 0:
        raise argparse.ArgumentTypeError("The parsed list shoud not be empty".format(s))

    return lst


def delimited_ints(s):
    return delimited_num(s, int)


def delimited_floats(s):
    return delimited_num(s, float)


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