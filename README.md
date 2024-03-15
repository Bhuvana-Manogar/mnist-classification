# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
![Screenshot 2024-03-15 200716](https://github.com/Bhuvana23013531/mnist-classification/assets/147125678/c08159b9-acbb-4a91-b81d-1633ba9c1ba1)

## Neural Network Model

![Screenshot 2024-03-15 201015](https://github.com/Bhuvana23013531/mnist-classification/assets/147125678/7be880e0-79da-47fb-b713-c4effa31bffd)


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:
Build a CNN model
### STEP 3:
Compile and fit the model and then predict

## PROGRAM

### Name: BHUVANESHWARI M
### Register Number:212223230033
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model.add (layers. Input (shape=(28,28,1)))
model.add (layers.Conv2D (filters=32, kernel_size=(3,3), activation='relu'))
model.add (layers.MaxPool2D (pool_size=(2,2)))
model.add (layers. Flatten())
model.add (layers.Dense (32, activation='relu'))
model.add (layers.Dense (10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('five.png')
type(img)
img = image.load_img('five.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
```
### OUTPUT
## Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-03-15 201355](https://github.com/Bhuvana23013531/mnist-classification/assets/147125678/f32f097f-345c-4239-af07-bbaf126c93c2)

![Screenshot 2024-03-15 201405](https://github.com/Bhuvana23013531/mnist-classification/assets/147125678/9c430366-1048-4858-8d8d-3b62b294b829)


### Classification Report

![Screenshot 2024-03-15 201430](https://github.com/Bhuvana23013531/mnist-classification/assets/147125678/c34678e4-b99a-43c6-abf8-e02e81e76286)

### Confusion Matrix
![Screenshot 2024-03-15 201446](https://github.com/Bhuvana23013531/mnist-classification/assets/147125678/b06ecfdc-ca9c-400c-89a4-6e321244fdcd)

### New Sample Data Prediction

![Screenshot 2024-03-15 201505](https://github.com/Bhuvana23013531/mnist-classification/assets/147125678/2abaedcb-8307-4609-ae21-f22ced68b514)

## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
