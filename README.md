# IndianFoodClassification

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import cv2
import numpy as np
import os
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import time

trainingDataset = []
img_size = 100
path = "/content/drive/My Drive/CV_Project_2/Train2"
classNumber = 0
trainingDataset.clear()

for folder in (os.listdir(path)):
  print(classNumber)
  print("Folder Name:",folder)
  # folder = with_mask ,without_mask
  fp = os.path.join(path,folder)
  # joining folder like /content/Face_Mask/Train/with_mask
  for eachImage in os.listdir(fp):
    imagePath = os.path.join(fp,eachImage)
    img = (cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE))/255
    resize=cv2.resize(img,(img_size,img_size))
    trainingDataset.append([resize,classNumber])
  classNumber = classNumber + 1
  
  
X = []
Y = []
img_size = 100
np.random.shuffle(trainingDataset)
for features, label in trainingDataset:
    X.append(features)
    Y.append(label)
print(Y) 
  
  
X = np.array(X).reshape(-1, img_size, img_size, 1)
Y_binary = to_categorical(Y)
print(X)
print(Y_binary)

model = Sequential()

model.add(Conv2D(40, (3, 3), input_shape=(100,100,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(60, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(80, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
 
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )
              
history = model.fit(X, Y_binary,
          batch_size = 32,
          epochs=20, validation_split = 0.1)
 
model.save("/content/drive/My Drive/CV_Project_2/Models/{NAME}.model")

plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

# summarize history for loss

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.tight_layout()

plt.show()

def prepare(filepath):
    img_size = 100 
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)/255  
    img_resize = cv2.resize(img, (img_size, img_size))  
    return img_resize.reshape(-1, img_size, img_size, 1)
    
    
prediction = model.predict(prepare("/content/drive/My Drive/CV_Project_2/Test/pizza.jpg"))
print((prediction))

# CATEGORIES = ["ButterNaan", "Burger","Chapati","Chai","Samosa","Pizza"]
CATEGORIES = ["Burger", "ButterNaan","Chai","Chapati","Pizza","Samosa","Dhokla","Jalebi","Kulfi","Paani Puri"]

pred_class = CATEGORIES[np.argmax(prediction)]
print(pred_class)

