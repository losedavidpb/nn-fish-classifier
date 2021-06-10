# --------------------------------------------
# Utilities for current neural network.
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

import abc
import numpy as np

from PIL import Image
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizer_v1 import SGD
from utils import list_files, get_parent

class ModelDefinition(abc.ABC):
    """Define new sequential models for neural networks. """

    @abc.abstractmethod
    def define(self, num_classes, target_size, **kwargs):
        """
        Return a new sequential neural network model with passed arguments.

        >> Available arguments

            * num_classes: number of classes that will be classify (necessary).
            * target_size: specific size that each image classified will has (necessary).
            * loss: loss function used during training compilation (default: mse).
            * optimizer: optimizer used during training compilation (default: SGD(lr=0.1)).
            * metrics: metrics used during training compilation (default: ['accuracy']).
        """
        pass

# _________________ Versions of neural network models for fish classification _________________

class ModelDefinitionLeakyRelu(ModelDefinition):
    def define(self, num_classes, target_size, **kwargs):
        loss = kwargs.get('loss', 'mse')
        optimizer = kwargs.get('optimizer', SGD(lr=0.1))
        metrics = kwargs.get('metrics', ['accuracy'])

        input_shape1, input_shape2 = target_size

        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(input_shape1, input_shape2, 3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        return model

class ModelDefinitionLeakyReluMoreConv(ModelDefinition):
    def define(self, num_classes, target_size, **kwargs):
        loss = kwargs.get('loss', 'mse')
        optimizer = kwargs.get('optimizer', SGD(lr=0.1))
        metrics = kwargs.get('metrics', ['accuracy'])

        input_shape1, input_shape2 = target_size

        model = Sequential()
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(input_shape1, input_shape2, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        return model

class ModelDefinitionTwoDense(ModelDefinition):
    def define(self, num_classes, target_size, **kwargs):
        loss = kwargs.get('loss', 'mse')
        optimizer = kwargs.get('optimizer', SGD(lr=0.1))
        metrics = kwargs.get('metrics', 'accuracy')

        input_shape1, input_shape2 = target_size

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(input_shape1, input_shape2, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        return model

def define_model(num_classes, target_size=(150, 150), **kwargs):
    """Return the best neural network model founded for fish classification. """
    return ModelDefinitionLeakyReluMoreConv().define(num_classes, target_size, **kwargs)

# ________________________ Training and testing ________________________

def train_model(model, train_generator, validation_generator, **kwargs):
    """Train passed model with EarlyStopping and passed arguments.

    >> Available arguments

        * model: compiled neural network model (necessary)
        * train_generator: ImageDataGenerator for training (necessary)
        * validation_generator: ImageDataGenerator for validation (necessary)
        * epochs: maximum number of epochs (default: 150)
        * patience: patience to wait for EarlyStopping (default: 3)
        * steps_per_epoch: steps for each epoch executed (default: 40)
    """
    epochs = kwargs.get('epochs', 150)
    patience = kwargs.get('patience', 3)
    steps_per_epoch = kwargs.get('steps_per_epoch', 40)

    es = EarlyStopping(
        monitor='val_accuracy', mode='max',
        verbose=1, patience=patience, restore_best_weights=True
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[es]
    )

    return history

def test_model(model, test_generator):
    """Test passed model using a ImageDataGenerator testing instance. """
    test_images = list_files('./dataset/test', mode='only_files')

    input_shape1, input_shape2 = test_generator.target_size
    classes = test_generator.class_indices
    num_total, num_errors = len(test_images), 0

    for image_dir in test_images:
        img = Image.open(image_dir, 'r').convert('RGB')
        img = np.asarray(img.resize((input_shape1, input_shape2)))
        img = img.reshape(1, input_shape1, input_shape2, 3)

        prediction = model.predict(img).flatten().tolist()
        class_num = prediction.index(max(prediction))
        expected = classes.get(get_parent(image_dir))

        for class_name in classes.keys():
            if classes.get(class_name) == class_num:
                if class_num != expected: num_errors += 1

                str_format = "Image<{}>: It's a {}"
                print(str.format(str_format, image_dir, class_name), sep="")
                break

    num_correct = num_total - num_errors

    print("Correct: ", num_correct)
    print("Error: ", num_errors)
    print("Total: ", num_total)
