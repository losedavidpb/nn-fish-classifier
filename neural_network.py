# --------------------------------------------
# Utilities for current neural network.
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

import abc

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizer_v1 import SGD

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
        model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(input_shape1, input_shape2, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128,kernel_size=(2, 2), activation='relu'))
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
