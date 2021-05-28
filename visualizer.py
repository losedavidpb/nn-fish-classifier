# --------------------------------------------
# Visualizer for a neural network.
# All Copyright reserved (c) based on LICENSE.
# --------------------------------------------

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from utils import list_files

def show_dataset_files():
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
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.xticks([i for i in range(0, len(history.history['accuracy']))])
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='upper left')

    plt.show()
