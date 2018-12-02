import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import math  
import cv2  
 
# This will train the top layer of the neural network using 
# bottleneck features extracting from the original training set

img_width, img_height = 224, 224  
# dimensions of our images.  
top_model_weights_path = 'dog_recognition_bottleneck_model.h5'  
train_data_dir = 'dogImages/train'  
validation_data_dir = 'dogImages/valid'  
# number of epochs to train top model  
epochs = 50  
# batch size used by flow_from_directory and predict_generator  
batch_size = 16 
num_classes = 136

datagen_top = ImageDataGenerator(rescale=1./255)  
generator_top = datagen_top.flow_from_directory(  
        train_data_dir,  
        target_size=(img_width, img_height),  
        batch_size=batch_size,  
        class_mode='categorical',  
        shuffle=False)  

nb_train_samples = len(generator_top.filenames)  
num_classes = len(generator_top.class_indices)  

# load the bottleneck features saved earlier  
train_data = np.load('bottleneck_features_train.npy')  

# get the class lebels for the training data, in the original order  
train_labels = generator_top.classes  

# convert the training labels to categorical vectors  
train_labels = to_categorical(train_labels, num_classes=num_classes)

generator_top = datagen_top.flow_from_directory(  
        validation_data_dir,  
        target_size=(img_width, img_height),  
        batch_size=batch_size,  
        class_mode=None,  
        shuffle=False)  

nb_validation_samples = len(generator_top.filenames)  

validation_data = np.load('bottleneck_features_validation.npy')  

validation_labels = generator_top.classes  
validation_labels = to_categorical(validation_labels, num_classes=num_classes) 

#definition of network
top_model = Sequential()  
top_model.add(Flatten(input_shape=train_data.shape[1:]))  
top_model.add(Dense(133, activation='relu'))  
top_model.add(Dropout(0.5))  
top_model.add(Dense(num_classes, activation='relu'))

top_model.compile(optimizer='rmsprop',  
            loss='categorical_crossentropy', metrics=['accuracy'])  

history = top_model.fit(train_data, train_labels,  
        epochs=epochs,  
        batch_size=batch_size,  
        validation_data=(validation_data, validation_labels))  

top_model.save_weights(top_model_weights_path)  

(eval_loss, eval_accuracy) = top_model.evaluate(  
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
print("[INFO] Loss: {}".format(eval_loss)) 