# --------------------------------------------
# Visualizer for a neural network.
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

import cv2 as cv
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import list_files

def show_dataset_files():
    """Show all files from current dataset and plot one of them. """
    train_files = list_files('./dataset/train/', mode='only_files')
    test_files = list_files('./dataset/test/', mode='only_files')

    img1 = cv.imread(train_files[np.random.randint(0, len(train_files))])
    img2 = cv.imread(test_files[np.random.randint(0, len(test_files))])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Examples from current Dataset")
    ax1.imshow(img1, cmap="gray")
    ax2.imshow(img2, cmap="gray")
    fig.show()

    print("===== TRAINING DATASET")
    for file in train_files: print(file)

    print("===== VALIDATION DATASET")
    for file in test_files: print(file)

def plot_history(history):
    """Plot the accuracy and loss evolution using passed history. """
    fig = plt.figure()

    plt.title('Fish Classifier Evolution')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.xticks([i for i in range(0, len(history.history['accuracy']))])
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['loss'], label='Loss value')
    plt.legend(loc='upper left')

    fig.savefig('plot_accuracy_loss.jpg', bbox_inches='tight', dpi=250)
    plt.show()

def plot_confusion_matrix(model, test_generator):
    """Plot the confusion matrix of current model using ImageDataGenerator testing instance. """
    test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)

    y_expected = test_generator.classes
    y_prediction = model.predict(test_generator, test_steps_per_epoch)
    y_prediction = np.argmax(y_prediction, axis=1)

    labels = test_generator.class_indices

    cm = confusion_matrix(y_expected, y_prediction, normalize='all')
    sns.heatmap(cm, annot=True, cmap="Spectral")

    plt.title('Confusion Matrix of Fish Classifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.xticks([i for i in range(0, len(labels))], labels=labels, rotation=20)
    plt.yticks([i for i in range(0, len(labels))], labels=labels, rotation=20)

    plt.savefig('plot_confusion_matrix.jpg', bbox_inches='tight', dpi=250)
    plt.show()
