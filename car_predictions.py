import numpy as np
import os
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, AveragePooling2D

# Load train & test files
files_train = 0
files_validation = 0
cwd = os.getcwd()
folder = 'train_data/train'

for sub_folder in os.listdir(folder):
   path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
   files_train += len(files)

folder = 'train_data/test'
for sub_folder in os.listdir(folder):
   path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
   files_train += len(files)

print(files_train, files_validation)

#set key parameters

img_width, img_height = 48, 48
train_data_dir = 'train_data/train'
validation_data_dir = 'train_data/test'
no_train_sample = files_train
no_validation_sample = files_validation
batch_size = 32
epochs = 10
num_classes = 2

#Build model on top of pretrained VGG

model = applications.VGG16(weights ='imagenet',include_top=False, input_shape=(img_width, img_height, 3))
model.summary()
model.layers

for layer in model.layers[:10]:
   layer.trainable = False

x = model.output
x = Flatten()(x)

predictions = Dense(num_classes, activation= 'softmax')(x)
model_final = Model(inputs = model.input, outputs = predictions)

model_final.compile(loss = 'categorical_crossentropy',
                    optimizer = optimizers.SGD(lr = 0.0001, momentum= 0.9),
                    metrics = ['accuracy'])

#Data Augmentation

train_datagen = ImageDataGenerator(rescale = 1.0/255, horizontal_flip = True, fill_mode='nearest', zoom_range=0.1,
                   width_shift_range=0.1,height_shift_range=0.1, rotation_range = 5)

test_datagen = ImageDataGenerator(rescale = 1.0/255, horizontal_flip = True, fill_mode='nearest', zoom_range=0.1,
                   width_shift_range=0.1,height_shift_range=0.1, rotation_range = 5)

# Create the training generator with data augmentation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # Shuffle the data
)

# Create the validation generator without data augmentation
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # Shuffle the data
)

# Optional: Calculate the number of steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Fit the model
history = model_final.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=epochs
)

history.history

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train','test'],loc = 'upper left')
plt.show()

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train','test'],loc = 'upper left')
plt.show()

model_final.save('model_final.h5')
class_dictionary = {}
class_dictionary[0] = 'no_car'
class_dictionary[1] = 'car'
class_dictionary

import cv2
import numpy as np

def make_prediction(image):
    image = cv2.imread(image)
    image = cv2.resize(image,(48, 48))
    img = image/255
#   (1, 48, 48, 3) -> 4D Tensor
    img = np.expand_dims(img, axis = 0)
    class_predicted = model_final.predict(img)
    intId = np.argmax(class_predicted[0])
    label = class_dictionary[intId]
    return label
make_prediction("roi_2.png")
make_prediction("spot124.png")
make_prediction("spot169.png")
model_final.save("model_final.h5")