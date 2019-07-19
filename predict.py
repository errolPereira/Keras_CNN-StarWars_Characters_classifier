from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

#loding the model saved previously
model = load_model('ResNet50_Star.h5')

#compiling the model
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

#Reading the test image
img = cv2.imread('Keras_CNN-StarWars_Characters_classifier/test.jpg')
#plotting the image
plt.imshow(img)
#resizing and reshaping the image to match out input
img = cv2.resize(img,(150,150))
img = np.reshape(img,[1,150,150,3])

#outputs a list with probabilities, just as a softmax function would return
classes = model.predict(img)

#getting the position of the highest probability class. That will be our predicted class
out = np.argmax(classes)


# Mapping position of the predicted class to the actual class itself. 
# The sequence is same as in our train image folder ie : 
# 0 - Chewbacca
# 1 - Dart Vader
# 2 - Yoda
if out == 0:
  print('Predicted Class: Chewbacca')
if out == 1:
  print('Predicted Class: Darth Vader')
if out == 2:
  print('Predicted Class: Yoda')
