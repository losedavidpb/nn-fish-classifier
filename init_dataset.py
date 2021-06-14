# --------------------------------------------
# Dataset Initializer for a new neural network.
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

from keras.preprocessing.image import ImageDataGenerator
from utils import list_files, make_directory
from utils import get_filename, remove_files

import random
import shutil

def prepare_dataset(classes, porc_train, porc_test):
    """Prepare a new dataset selecting files for train and test randomly. """
    class_dirs = list_files('./dataset/', mode='only_dirs', recursion=False)
    class_dirs = [f for f in class_dirs if get_filename(f) not in ['train', 'test']]
    if len(class_dirs) != len(classes) or len(classes) <= 0: return

    remove_files('./dataset/train'), remove_files('./dataset/test')
    make_directory('./dataset/train'), make_directory('./dataset/test')

    verbose_format = "Preparing {} for class {} ........ "

    for class_name in classes:
        class_train_path = './dataset/train/' + class_name
        class_test_path = './dataset/test/' + class_name

        # Preparing train and test folders
        remove_files(class_train_path), make_directory(class_train_path)
        remove_files(class_test_path), make_directory(class_test_path)

        # Files will be shuffled in order to get a different dataset
        files = list_files('./dataset/' + class_name, mode='only_files')
        random.shuffle(files)

        num_train = int((porc_train * len(files)) / 100)
        num_test = int((porc_test * len(files)) / 100)

        # Copying all files to train and test folders
        print(str.format(verbose_format, 'training', class_name), end="")
        __copy_file(files, num_train, class_train_path)
        print("OK")

        print(str.format(verbose_format, 'validation', class_name), end="")
        __copy_file(files, num_test, class_test_path)
        print("OK")

        # Deleting old dataset to reduce memory use
        remove_files('./dataset/' + class_name)

def restore_dataset(classes):
    """Restore current dataset to the original. """
    class_dirs = list_files('./dataset/', mode='only_dirs', recursion=False)
    class_dirs = [f for f in class_dirs if get_filename(f) in ['train', 'test']]
    if len(class_dirs) != 2: return

    verbose_format = "Restoring class {} ........ "

    for class_name in classes:
        class_train_path = './dataset/train/' + class_name
        class_test_path = './dataset/test/' + class_name

        make_directory('./dataset/' + class_name + '/')

        train_files = list_files(class_train_path, mode='only_files')
        test_files = list_files(class_test_path, mode='only_files')

        print(str.format(verbose_format, class_name), end="")
        __copy_file(train_files, len(train_files), './dataset/' + class_name + '/')
        __copy_file(test_files, len(test_files), './dataset/' + class_name + '/')
        print("OK")

        remove_files(class_train_path)
        remove_files(class_test_path)

def init_train_generator(**kwargs):
    """Initialize a new ImageDataGenerator instance for training.

    >> Available arguments

        * batch_size: batch size for images loaded (default: 20)
        * class_mode: class mode, generally categorical used (default: binary)
        * target_size: each image loaded will has target_size with and height (default: (150, 150))
        * rescale: rescale used for data augmentation (default: 1./255)
        * rotation_range: rotation range used for data augmentation (default: 10)
        * zoom_range: zoom range used for data augmentation (default: 0.5)
        * width_shift_range: width shift range used for data augmentation (default: 0.2)
        * height_shift_range: height shift range used for data augmentation (default: 0.2)
        * brightness_range: brightness range used for data augmentation (default: (0.2,0.8))
        * horizontal_flip: images will be flipped horizontally whether is true (default: False)
        * vertical_flip: images will be flipped vertically whether is true (default: False)
        * shuffle: images will be shuffled whether is true (default: True)
    """
    batch_size = kwargs.get('batch_size', 20)
    class_mode = kwargs.get('class_mode', 'binary')
    shuffle = kwargs.get('shuffle', False)

    target_size = kwargs.get('target_size', (150, 150))
    rescale = kwargs.get('rescale', 1. / 255)

    rotation_range = kwargs.get('rotation_range', 10)
    zoom_range = kwargs.get('zoom_range', 0.5)
    width_shift_range = kwargs.get('width_shift_range', 0.2)
    height_shift_range = kwargs.get('height_shift_range', 0.2)
    brightness_range = kwargs.get('brightness_range', (0.2, 0.8))

    horizontal_flip = kwargs.get('horizontal_flip', False)
    vertical_flip = kwargs.get('vertical_flip', False)

    train_data_gen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_range=brightness_range
    )

    return train_data_gen.flow_from_directory(
        './dataset/train',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle
    )

def init_test_generator(**kwargs):
    """Initialize a new ImageDataGenerator instance for testing.

        >> Available arguments

            * batch_size: batch size for images loaded (default: 20)
            * class_mode: class mode, generally categorical used (default: binary)
            * target_size: each image loaded will has target_size with and height (default: (150, 150))
            * rescale: rescale used for data augmentation (default: 1./255)
            * shuffle: images will be shuffled whether is true (default: True)
        """
    target_size = kwargs.get('target_size', (150, 150))
    rescale = kwargs.get('rescale', 1. / 255)
    batch_size = kwargs.get('batch_size', 20)
    class_mode = kwargs.get('class_mode', 'binary')
    shuffle = kwargs.get('shuffle', True)

    test_data_gen = ImageDataGenerator(rescale=rescale)

    return test_data_gen.flow_from_directory(
        './dataset/test',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle
    )

# _____________________ Private functions _____________________

# Copy passed files to defined path but only num_files files
def __copy_file(files, num_files, new_path):
    for _ in range(0, num_files):
        if len(files) == 0: break

        dataset_file = files.pop()
        dataset_filename = get_filename(dataset_file)
        shutil.copyfile(dataset_file, new_path + '/' + dataset_filename)

    return files
