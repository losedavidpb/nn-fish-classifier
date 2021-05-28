# --------------------------------------------
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

from init_dataset import prepare_dataset, init_train_generator
from init_dataset import init_test_generator, delete_dataset
from visualizer import show_dataset_files, plot_history
from neural_network import define_model, train_model, test_model
from utils import list_files, get_parent
import tensorflow as tf

def _get_classes(path_classes):
    list_of_files = list_files(path_classes, recursion=False)
    nn_list_classes = []

    for filename in list_of_files:
        class_name = get_parent(filename)

        if class_name not in nn_list_classes:
            if class_name != 'train' and class_name != 'test':
                nn_list_classes.append(class_name)

    return nn_list_classes

# ============== GENERAL CONFIGURATION ================

# -- Basic neuronal network config
classes = _get_classes('./dataset')
num_classes = len(classes)
model_dir = "nn_model.h5"

# -- Neural network training config
epochs = 20
steps_per_epoch = 70
batch_size = 30
patience = 3

# -- Neural network compilation config
loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adadelta()
metrics = 'accuracy'

# -- Dataset splitter config
porc_train = 80
porc_test = 100 - porc_train

# -- Dataset initializer config
class_mode = 'categorical'
target_size = (200, 200)
rescale = 1./255
rotation_range = 30
zoom_range = 0.7
width_shift_range = 0.1
height_shift_range = 0.1

if __name__ == '__main__':
    prepare_dataset(classes, porc_train, porc_test)
    # show_dataset_files()

    train_generator = init_train_generator(
        target_size=target_size,
        rescale=rescale,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        class_mode=class_mode,
        batch_size=batch_size,
    )

    test_generator = init_test_generator(
        target_size=target_size,
        rescale=rescale,
        class_mode=class_mode,
        batch_size=batch_size
    )

    model = define_model(
        num_classes=num_classes, target_size=target_size,
        loss=loss, optimizer=optimizer, metrics=metrics
    )

    history = train_model(
        model, train_generator, test_generator,
        epochs=epochs, steps_per_epoch=steps_per_epoch, patience=patience
    )

    test_model(model, test_generator)
    plot_history(history)
    delete_dataset(classes)

    answer = input("Do you want to save current model? ")

    while answer != 'y' and answer != 'n':
        answer = input("Do you want to save current model? ")

    if answer == 'y': model.save(model_dir)
