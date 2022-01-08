import os
from argparse import ArgumentTypeError


def join(a, b):
    """
    A custom path join function which replaces "\\" with "/". Otherwise,
    os.path.join may cause problems with filepaths (especially on Windows).

    Parameters:
        a (str): Start of file path.
        b (str): End of file path.
    
    Returns:
        (str): The joined path of a and b.
    """
    return os.path.join(a, b).replace("\\", "/")


# Define relative paths
SRC_PATH = str(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
ROOT_PATH = os.path.dirname(SRC_PATH)
VISUALIZE_PATH = join(ROOT_PATH, 'visualizations')
MODEL_PATH = join(ROOT_PATH, 'models')
DATA_PATH = join(ROOT_PATH, 'data')


# Define argument parsing types

# Paths
def path(path):
    """
    Argparsing type: Path to an existing file or directory.

    Parameters:
        path (str): Path.
    
    Returns:
        (str): Validated path.
    """
    if os.path.isdir(path) or os.path.isfile(path):
        return path
    else:
        raise ArgumentTypeError("{} is not a valid file or directory path".format(path))


def file_path(path):
    """
    Argparsing type: Path to an existing file.

    Parameters:
        path (str): Path.
    
    Returns:
        (str): Validated file path.
    """
    if os.path.isfile(path):
        return path
    else:
        raise ArgumentTypeError("{} is not a valid file path".format(path))


def dir_path(path):
    """
    Argparsing type: Path to an existing directory.

    Parameters:
        path (str): Path.
    
    Returns:
        (str): Validated directory path.
    """
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError("{} is not a valid directory path".format(path))


def save_path(path):
    """
    Argparsing type: Path to which one can save a file or directory.
    This path need not exist, but the parent directory must.

    Parameters:
        path (str): Path.
    
    Returns:
        (str): Validated save path.
    """
    if os.path.isdir(os.path.dirname(path)):
        return path
    else:
        ArgumentTypeError("{} is not a valid directory path to save a file".format(os.path.dirname(path)))


def save_file_path(path):
    """
    Argparsing type: Path to which one can save a file.
    This path need not exist, but the parent directory must.

    Parameters:
        path (str): Path.
    
    Returns:
        (str): Validated file save path.
    """
    return save_path(path)


def save_image_path(path):
    """
    Argparsing type: Path to which one can save an .png or .jpg file.
    This path need not exist, but the parent directory must.

    Parameters:
        path (str): Path.
    
    Returns:
        (str): Validated image save path.
    """
    if path[-4:] in ['.png', '.jpg']:
        return save_file_path(path)
    else:
        raise ArgumentTypeError("Expected file extensions '.png' or '.jpg' for {}".format(path))


def save_dir_path(path):
    """
    Argparsing type: Path to which one can save files in a directory.
    This path must be to an existing directory, or at least its parent must
    exist and the directory will be created.

    Parameters:
        path (str): Path.
    
    Returns:
        (str): Validated directory save path.
    """
    # Return if it exists
    if os.path.isdir(path):
        return path

    # Create directory and return if its parent exists
    if os.path.isdir(os.path.dirname(path)):
        os.mkdir(path)
        return path

    else:
        raise ArgumentTypeError("{} directory must exist".format(os.path.dirname(path)))


# Delimited types
def delimited_num(s, num_type):
    """
    A base function off which to create comma-delimited number argparsing types.

    Parameters:
        s (str): Delimited numbers in string format, e.g., '(10, 15, 9)'.
        num_type (class): A class to convert individual numbers, e.g., int or float.
    
    Returns:
        (list<num_type>): A list with values of type num_type.
    """
    if not isinstance(s, str):
        raise ArgumentTypeError("{} must be a string".format(s))
    
    # Keep only numbers, commas, and periods
    import re
    s = re.sub("[^0-9.,-]", "", s)
    
    # Split and convert to integers
    try:
        lst = [num_type(i) for i in s.split(",")]
    except:
        raise ArgumentTypeError("Cannot convert {} to a list of type {}".format(s, num_type.__name__))

    # Check whether empty
    if len(lst) == 0:
        raise ArgumentTypeError("The parsed list shoud not be empty".format(s))

    return lst


def delimited_ints(s):
    """
    Argparsing type: Comma-delimited integers. Brackets and spacing are irrelevant.
    Errors will be thrown if the individual values cannot be converted to integers.

    Parameters:
        s (str): Delimited integers in string format, e.g., '(10, -15, 9)'.
    
    Returns:
        (list<float>): A list of integers.
    """
    return delimited_num(s, int)


def delimited_floats(s):
    """
    Argparsing type: Comma-delimited floats. Brackets and spacing are irrelevant.
    Errors will be thrown if the individual values cannot be converted to floats.

    Parameters:
        s (str): Delimited floats in string format, e.g., '(10.5, -15.9, -9)'.

    Returns:
        (list<float>): A list of floats.
    """
    return delimited_num(s, float)


# Other types
def percent(p):
    """
    Argparsing type: A percentage. Handles both 0-1 inputs, and 1-100 inputs.
    If not in one of these forms, an error is thrown.

    Parameters:
        p (str): Percentage string.

    Returns:
        (float): A decimal percentage.
    """
    try:
        p = float(p)
    except:
        return ArgumentTypeError("{} is not a valid percentage".format(p))
    
    if p < 0:
        return ArgumentTypeError("{} is not a valid percentage".format(p))
    # Check whether a decimal percentage was inputed
    if p > 1:
        if p > 100:
            return ArgumentTypeError("{} is not a valid percentage".format(p))
        return p/100
    return p