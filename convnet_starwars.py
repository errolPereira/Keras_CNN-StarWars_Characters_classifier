import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

################################### Image Directory ###############################
#base directory
base_dir = r'C:\Users\errperei\Desktop\HPE\Errol_docs\Python Exercises\GIT REPO\Keras_CNN-StarWars_Characters_classifier\Star wars Images'

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

#validation darth vader
validation_yoda_dir = os.path.join(validation_dir, 'yoda')

#validation chewbacca 
validation_chewbacca_dir = os.path.join(validation_dir, 'chewbacca')

############################loading the images name ###############################
#darth vader train image names
train_darth_fnames = os.listdir(train_darth_vader_dir)
print('Darth Vader train Images: {}'.format(len(train_darth_fnames)))

#yoda train image names
train_yoda_fnames = os.listdir(train_yoda_dir)
print('Yoda train Images: {}'.format(len(train_yoda_fnames)))

#Chewbacca train image names
train_chewbacca_fnames = os.listdir(train_chewbacca_dir)
print('Chewbacca train Images: {}'.format(len(train_chewbacca_fnames)))

######################### displaying Images ###################################
#some constants for the training
IMAGE_SIZE = (200, 200)
batch_size = 100
EPOCHS = 5

plt.figure(0, figsize=(12, 20))
cpt = 0

for characters in os.listdir(train_dir):
    for i in range(1, 4):
        cpt = cpt + 1
        plt.subplot(3, 3, cpt)
        img = load_img(os.path.join(train_dir, characters) + "/" + os.listdir(train_dir +"/"+ characters)[i], target_size=(200, 200))
        plt.imshow(img, cmap='gray')
        
plt.tight_layout()
plt.show()


########################### Setup Image Generators ################################
train_datagen = ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                vertical_flip=True,
                width_shift_range=0.2,
                shear_range=0.2,
                height_shift_range=0.2)

#No augmentation used for the test dataset
test_datagen = ImageDataGenerator()

#Creating the train data generator
train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=IMAGE_SIZE,
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=True
                    )
#creating the validation data generator
test_generator = test_datagen.flow_from_directory(
                    validation_dir,
                    target_size=IMAGE_SIZE,
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=True
                    )

########################### COnvolution Layers #####################################
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

#number of possible label values
num_labels = 3

#input layer
img_input = Input(shape=(200, 200, 3))

#1-Convolution
x = Conv2D(64, (3, 3), input_shape=(200, 200, 1), padding='same')(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

#2-Convolution
x = Conv2D(128, (5, 5), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

#3-Convolution
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

#4-Convolution
x = Conv2D(512, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

######################### Fully connected layuers #############################

#flattening to a 1D vector
x = Flatten()(x)

#Fully connected 1st layer
x = Dense(256)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)

#Fully connected 2nd layer
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)

output = Dense(num_labels, activation='softmax')(x)

#initializing the optimizerr
opt = Adam(0.0001)


#Initializing the model
model = Model(img_input, output)
#configuring the model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['acc'])

########################## Training the model #####################################
from tensorflow.keras.callbacks import ModelCheckpoint

#Creating a checkpoint and saving the model weights
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#training the model
history = model.fit_generator(generator=train_generator,
                              steps_per_epoch = 100,
                              epochs = EPOCHS,
                              validation_data = test_generator,
                              validation_steps = test_generator.n//test_generator.batch_size,
                              verbose=2,
                              callbacks=callbacks_list)

########################### Analyzing the results #################################
#Retrieving the accuracy and loss from the model at each epoch
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

#plotting the acc vs val_acc plot
epochs = len(acc)

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()
