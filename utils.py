# --------------------------------------------
# Utilities for current neural network.
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

import os

def make_directory(directory_path):
    """Make a new directory with passed path. """
    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

def get_filename(file_path):
    """Get filename of passed path. """
    split_path = file_path.split("/")
    return split_path[len(split_path) - 1]

def get_parent(file_path):
    """Get parent path of passed path. """
    split_path = file_path.split("/")
    return split_path[len(split_path) - 2]

def remove_file(file_path):
    """Remove file at passed file. """
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            os.removedirs(file_path)

def remove_files(directory_path):
    """Remove files that are located at passed path. """
    if os.path.exists(directory_path):
        for file in list_files(directory_path):
            remove_file(file)

        remove_file(directory_path)

def list_files(dirname, mode='default', recursion=True):
    """List files that are located at passed directory.

    It is possible to specify what files will be included
    at last result, so you can include all files founded,
    or just regular files or directories.

    >> Available modes:

        * default: list files but also directories
        * only_files: list files that are not directories
        * only_dirs: list directories but files are avoided

    If recursion parameter is True, this function will try to
    search for other files at subdirectories.
    """
    if recursion is True:
        return __list_files(mode, dirname, [])

    files = os.listdir(dirname)

    # Check if list will have all files
    if mode == 'default': return files

    # Check filter mode in order to define output
    if mode == 'only_dirs':
        return [f for f in files if os.path.isdir(dirname + '/' + f)]
    elif mode == 'only_files':
        return [f for f in files if os.path.isfile(dirname + '/' + f)]

# ______________________ Private functions ______________________

# Return files located at passed directory using recursion
def __list_files(_mode, _dirname, _files):
    for _filename in os.listdir(_dirname):
        if _dirname[len(_dirname) - 1] == '/':
            _dirname = _dirname[:len(_dirname) - 1]

        _filename = _dirname + '/' + _filename

        if _mode == 'default' or _mode == 'only_files':
            if os.path.isfile(_filename):
                _files.append(_filename)
            elif os.path.isdir(_filename):
                _files = __list_files(_mode, _filename, _files)
                if _mode == 'default': _files.append(_filename)
        elif _mode == 'only_dirs':
            if os.path.isdir(_filename):
                _files.append(_filename)
                _files = __list_files(_mode, _filename, _files)

    return _files
