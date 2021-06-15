# NEURAL NETWORK FISH CLASSIFIER

## Introduction

This is an implementation of a convolutional neural network that is able to classify groups of fish.
The dataset has got four different fish classes, each one composed of 1000 images, and they have been
obtained from a Kaggle dataset (see https://www.kaggle.com/crowww/a-large-scale-fish-dataset).

## Results

* ### Accuracy and Loss Evolution

![Accuracy and Loss Evolution](images/plot_accuracy_loss.jpg "Accuracy and Loss Evolution")

* ### Confusion Matrix

![Confusion Matrix](images/plot_confusion_matrix.jpg "Confusion Matrix")

## Hyperparameters and Parameters

* ### Dataset initializer
  
| Name  | Definition | Value |
| ----  | ---------- | ----- | 
| Training percentage | Amount of files for training in % | 80% |
| Validation percentage | Amount of files for validation in % | 20% |

* ### Neural network Compilation

| Name  | Definition | Value |
| ----  | ---------- | ----- |
| Loss function | Loss function used during training | Categorical Cross Entropy |
| Optimizer | Optimizer used during training | Ada Delta |

* ### Training and EarlyStopping

| Name  | Definition | Value |
| ----  | ---------- | ----- |
| Number of Epochs | Maximum number of iterations | 30 |
| Steps per Epoch  | Total steps of each epoch executed | 70 |
| Batch size | Number of files loaded during training steps | 30 |
| Patience   | Maximum tries to improve val_accuracy before exit | 3 |

* ### Data Augmentation

| Name  | Definition | Value |
| ----  | ---------- | ----- |
| Target size | Width and height for each image loaded | 200w, 200h |
| Rescale | Rescale used during image loading process | 1./255 |
| Rotation range | Rotation range for data augmentation | 30 |
| Zoom range | Zoom range for data augmentation | 0.7 |
| Width shift range | Width swift range for data augmentation | 0.1 | 
| Height shift range | Width swift range for data augmentation | 0.1 |
| Brightness range | Brightness range for data augmentation | (0.2, 0.8) |
| Horizontal flip | Horizontal flip for data augmentation | True |
| Vertical flip | Vertical flip for data augmentation | True |

* ### Models

| Name | Structure |
| ---- | --------- |
| ModelDefinitionTwoDense | <ul><li>Conv2D(filters=32, kernel=(3, 3), activation=relu)</li> <li>MaxPooling((2,2))</li> <li>Dropout(0.25)</li> <li>Flatten()</li> <li>Dense(128, activation=relu)</li> <li>Dropout(0.25)</li> <li>Dense(64, activation=relu)</li> <lI>Dense(4, activation=softmax)</li></ul> |
| ModelDefinitionLeakyRelu | <ul><li>Conv2D(filters=128, kernel=(3, 3), activation=relu)</li> <li>LeakyReLU(alpha=0.1)</li> <li>MaxPooling((2,2))</li> <li>Dropout(0.25)</li> <li>Flatten()</li> <li>Dense(64, activation=relu)</li> <li>Dropout(0.5)</li> <li>Dense(32, activation=relu)</li> <li>Dense(4, activation=softmax)</li></ul> |
| ModelDefinitionLeakyReluMoreConv | <ul><li>Conv2D(filters=128, kernel=(3, 3), activation=relu)</li> <li>MaxPooling((2,2))</li> <li>Dropout(0.25)</li> <li>LeakyReLU(alpha=0.1)</li> <li>MaxPooling((2,2))</li> <li>Dropout(0.25)</li> <li>Flatten()</li> <li>Dense(64, activation=relu)</li> <li>Dropout(0.5)</li> <li>Dense(32, activation=relu)</li> <li>Dense(4, activation=softmax)</li></ul> |

| Name | Loss | Accuracy | Validation Loss | Validation Accuracy |
| ---- | ---- | -------- | --------------- | ------------------- |
| ModelDefinitionTwoDense | 1.3293 | 0.3835 | 1.0921 | 0.7097 |
| ModelDefinitionLeakyRelu | 1.2832 | 0.4655 | 1.0627 | 0.6625 |
| ModelDefinitionLeakyReluMoreConv  | 1.3145 | 0.3952 | 1.1267 | 0.6525 |

TwoConv: loss: 1.2537 - accuracy: 0.5282 - val_loss: 1.0885 - val_accuracy: 0.6775
OneConv: loss: 1.2742 - accuracy: 0.4593 - val_loss: 0.9729 - val_accuracy: 0.7000
ThreeConv: loss: 1.3756 - accuracy: 0.3024 - val_loss: 1.3326 - val_accuracy: 0.5450
2. loss: 1.2016 - accuracy: 0.4935 - val_loss: 0.8814 - val_accuracy: 0.7013   
3. loss: 1.3380 - accuracy: 0.3909 - val_loss: 1.2122 - val_accuracy: 0.6338

## Authors

- losedavidpb: [https://github.com/losedavidpb](https://github.com/losedavidpb)
- SergioULPGC: [https://github.com/SergioULPGC](https://github.com/SergioULPGC)
