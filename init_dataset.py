# --------------------------------------------
# Dataset Initializer for a new neural network.
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

from keras.preprocessing.image import ImageDataGenerator
from utils import list_files, make_directory, remove_file
from utils import get_filename, remove_files

import random
import shutil

def __copy_file(files, num_files, class_path):
    for _ in range(0, num_files):
        if len(files) == 0: break

        dataset_file = files.pop()
        dataset_filename = get_filename(dataset_file)
        shutil.copyfile(dataset_file, class_path + '/' + dataset_filename)

    return files

def prepare_dataset(classes, porc_train, porc_test):
    remove_files('./dataset/train')
    remove_files('./dataset/test')

    make_directory('./dataset/train')
    make_directory('./dataset/test')

    verbose_format = "Preparing {} for class {} ........ "

    for class_name in classes:
        class_train_path = './dataset/train/' + class_name
        class_test_path = './dataset/test/' + class_name

        # Preparing train and test folders
        remove_files(class_train_path)
        make_directory(class_train_path)

        remove_files(class_test_path)
        make_directory(class_test_path)

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

def init_train_generator(**kwargs):
    batch_size = kwargs.get('batch_size', 20)
    class_mode = kwargs.get('class_mode', 'binary')

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
        class_mode=class_mode
    )

def init_test_generator(**kwargs):
    target_size = kwargs.get('target_size', (150, 150))
    rescale = kwargs.get('rescale', 1. / 255)
    batch_size = kwargs.get('batch_size', 20)
    class_mode = kwargs.get('class_mode', 'binary')

    test_data_gen = ImageDataGenerator(rescale=rescale)

    return test_data_gen.flow_from_directory(
        './dataset/test',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode
    )

def delete_dataset(classes):
    for class_name in classes:
        remove_files('./dataset/train/' + class_name)
        remove_files('./dataset/test/' + class_name)

    remove_file('./dataset/train/')
    remove_file('./dataset/test/')
