# --------------------------------------------
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

from init_dataset import prepare_dataset, init_train_generator, restore_dataset
from init_dataset import init_test_generator, delete_dataset
from visualizer import plot_history, plot_confusion_matrix
from neural_network import define_model, train_model, test_model
from utils import list_files, get_parent
import tensorflow as tf
import keras.models

def _get_classes(path_classes):
    """Get current classes avoiding trash files. """
    list_of_files = list_files(path_classes, recursion=False)
    nn_list_classes = []

    for filename in list_of_files:
        class_name = get_parent(filename)

        if class_name not in nn_list_classes:
            if class_name != 'train' and class_name != 'test':
                nn_list_classes.append(class_name)

    return nn_list_classes

# ______________________ General configuration ______________________

classes = _get_classes('./dataset')
if len(classes) == 0: classes = _get_classes('./dataset/train')
num_classes = len(classes)
model_dir = "nn_model.h5"

# ______________________ Hyperparameters and parameters ______________________

# Dataset initializer
porc_train, porc_test = 80, 20

# Neural network Compilation
loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adadelta()
metrics = 'accuracy'

# Training and EarlyStopping
epochs = 30
steps_per_epoch = 40
batch_size = 30
patience = 3

# Data Augmentation
class_mode = 'categorical'
target_size = (250, 150)
rescale = 1. / 255
rotation_range = 30
zoom_range = 0.7
width_shift_range = 0.1
height_shift_range = 0.1
horizontal_flip = True
vertical_flip = True
brightness_range = (0.2, 0.8)

# ______________________ Main functions ______________________

def _init_generators():
    """Return ImageDataGenerator for training and testing. """
    train_generator = init_train_generator(
        target_size=target_size,
        rescale=rescale,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        class_mode=class_mode,
        batch_size=batch_size,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_range=brightness_range,
        shuffle=True
    )

    test_generator = init_test_generator(
        target_size=target_size,
        rescale=rescale,
        class_mode=class_mode,
        batch_size=batch_size,
        suffle=True
    )

    return train_generator, test_generator

def _prepare_dataset():
    """Prepare a new dataset with current hyperparameters. """
    prepare_dataset(classes, porc_train, porc_test)
    return _init_generators()

def _train_and_test(train_generator, test_generator):
    """Execute training and testing process. """
    if _ask_for("Do you want to load current model? "):
        model = keras.models.load_model(model_dir)
    else:
        model = define_model(
            num_classes=num_classes, target_size=target_size,
            loss=loss, optimizer=optimizer, metrics=metrics
        )

    history = train_model(
        model, train_generator, test_generator,
        epochs=epochs, steps_per_epoch=steps_per_epoch, patience=patience
    )

    test_model(model, test_generator, verbose_each_image=False)
    return model, history

def _visualize_results(history, model, test_generator):
    """Execute visualization process. """
    plot_history(history)
    plot_confusion_matrix(model, test_generator)

def _ask_for_action(question, action_true, action_false=None):
    """Execute passed action if user wants to. """
    if _ask_for(question): action_true()
    elif action_false is not None: action_false()

def _ask_for(question):
    """Check if user wants to do an action. """
    answer = input(question).lower()

    while answer not in ['y', 'n']:
        answer = input(question).lower()

    return answer == 'y'

def _ask_action_with_options(question, options_with_func):
    """Execute the action associated with option selected by user. """
    print("Available options")

    for opt in options_with_func.keys():
        print("\t*", opt)

    answer = input(question).lower()

    while answer not in options_with_func.keys():
        answer = input(question).lower()

    options_with_func[answer]()

# ______________________ Main ______________________

def _execute_for_train_test():
    if _ask_for("Do you want to delete current dataset? "):
        train_gen, test_gen = _prepare_dataset()
    else:
        train_gen, test_gen = _init_generators()

    model, history = _train_and_test(train_gen, test_gen)

    test_gen = init_test_generator(
        target_size=target_size,
        rescale=rescale,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False,
    )

    _visualize_results(history, model, test_gen)

    # Save current model if user wants
    _ask_for_action("Do you want to save current model? ", lambda: model.save(model_dir))

    # Cleaning dataset for next executions
    _ask_for_action("Do you want to delete current dataset? ", lambda: restore_dataset(classes))

def _execute_for_use():
    model = keras.models.load_model(model_dir)

    test_generator = init_test_generator(
        target_size=target_size,
        rescale=rescale,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=False,
    )

    # test_model(model, test_generator)
    plot_confusion_matrix(model, test_generator)

def _execute():
    _ask_action_with_options(
        "What do you want to do? ",
        options_with_func={
            'restore': lambda: restore_dataset(classes),
            'train': lambda: _execute_for_train_test(),
            'use': lambda: _execute_for_use()
        }
    )

if __name__ == '__main__': _execute()
