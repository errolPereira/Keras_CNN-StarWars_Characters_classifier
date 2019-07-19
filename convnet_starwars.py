# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:39:26 2019

@author: errperei
"""

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#loading the images
#base directory
base_dir = r'Keras_CNN-StarWars_Characters_classifier/Star wars Images'

#train_dir
train_dir = os.path.join(base_dir, 'train')

#validation directory
validation_dir = os.path.join(base_dir, 'validation')

#Darth Vader training images
train_darth_vader_dir = os.path.join(train_dir, 'darth vader')

#Yoda training images
train_yoda_dir = os.path.join(train_dir, 'yoda')

#Chewbacca train images
train_chewbacca_dir = os.path.join(train_dir, 'chewbacca')

#validation darth vader
validation_darth_dir = os.path.join(validation_dir, 'darth vader')

#validation yoda 
validation_yoda_dir = os.path.join(validation_dir, 'yoda')

#validation chewbacca 
validation_chewbacca_dir = os.path.join(validation_dir, 'chewbacca')

########################## Displaying some images ##############################
#loading the images name
#darth vader
train_darth_fnames = os.listdir(train_darth_vader_dir)
# print(train_darth_fnames[:10])
#yoda
train_yoda_fnames = os.listdir(train_yoda_dir)
# print(train_yoda_fnames[:10])
#chewbacca
train_chewbacca_fnames = os.listdir(train_chewbacca_dir)

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 5, nrows * 5)

pic_index += 8
next_darth_pix = [os.path.join(train_darth_vader_dir, fname) 
                for fname in train_darth_fnames[pic_index-8:pic_index]]
next_yoda_pix = [os.path.join(train_yoda_dir, fname) 
                for fname in train_yoda_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_darth_pix+next_yoda_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

####################################### Transfer Learning ###################################################

from keras.applications.resnet50 import ResNet50, preprocess_input

#Initializing the ResNet50 model and reusing it
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

#data genrators with image augmentation
train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=90,
                horizontal_flip=True,
                vertical_flip=True)

#test data should not be augmented
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

#train generator
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(150, 150),
                    batch_size=8)

#validation generator
validation_generator = train_datagen.flow_from_directory(
                    validation_dir,
                    target_size=(150, 150),
                    batch_size=8)

#################################### Fully Connected Dende Layers ################################
#Building top layers
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model

def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
    
    #output from the ResNet50 models last layer
    x = base_model.output
    
    #Flattening into 1D vector
    x = Flatten()(x)
    
    #looping over the number of fully connected layers
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        #dropouts
        x = Dropout(dropout)(x)
    
    #output layer
    predictions = Dense(num_classes, activation='sigmoid')(x)
    
    #creating a model
    finetune_model = Model(base_model.input, predictions)

    return finetune_model

################################# Training the model #################################################
#classes
class_list = ['chewbacca', 'darth_vader', 'yoda']
#number of hiiden layers and nodes
fc_layers = [512, 512]
#dropout percentage 
dropout = 0.5

#getting the model
finetune_model = build_finetune_model(base_model, dropout, fc_layers, len(class_list))

#optimizer for minimizing the loss 
from keras.optimizers import  Adam
adam = Adam(lr=0.00001)

#compiling/configuring the model
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

#training the model
history = finetune_model.fit_generator(train_generator, epochs=10, workers=8, 
                                        steps_per_epoch=100, 
                                        shuffle=True, 
                                        validation_data=validation_generator,
                                        validation_steps=50)
#################################### Plotting ############################################
def plot_training(history):
    #Retrieving the acc, loss, val_acc and val_loss from the model at each epoch
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    #number of epochs
    epochs = range(len(acc))
    
    #plotting Training vs Validation accuracy
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')

    plt.figure()
    #plotting Training vs Validation loss
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.show()

    #plt.savefig('acc_vs_epochs.png')
    
plot_training(history)

#Saving the model for future predictions
finetune_model.save('ResNet50_Star.h5')
