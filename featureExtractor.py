from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense  
from keras.models import Sequential  
from keras.preprocessing import image     
from keras.utils import np_utils
from tqdm import tqdm
import numpy as np
import math
import os
from glob import glob
from sklearn.datasets import load_files
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True
# Runs on test half of dogs we pulled from online, puts their bottleneck features in bottleneck_features_train.npy
#train_path = os.path.join("specificDogs","babyBeckham")
train_path = os.path.join("dogImages","train")
valid_path = os.path.join("dogImages","valid")
test_path = os.path.join("dogImages","test")

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 138)
    return dog_files, dog_targets

# load test dataset
train_files, train_labels = load_dataset(train_path)
train_bottleneck_features_path = 'train_bottleneck_features' #we need to put the generated weights into here for trainer.py
valid_bottleneck_features_path = 'valid_bottleneck_features'
test_bottleneck_features_path = "test_bottleneck_features"
# load list of dog names (this does not correspond to imagenet labels)
#dog_names = [item[20:-1] for item in sorted(glob(training_path))]
# num_classes = len(dog_names)

# define ResNet50 model
base_ResNet50 = ResNet50(include_top=False, weights='imagenet')
 
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    img = img / 255 # not sure if i need to do this, may remove later
    bottle_features = base_ResNet50.predict(img)
    return bottle_features

#for loop?
batchsize=16
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(train_path, target_size=(224,224),
                                        batch_size=batchsize, class_mode=None, shuffle=False)
train_samples = len(generator.filenames)
num_classes = len(generator.class_indices)
predict_size_train = int(math.ceil(train_samples/batchsize))

print("generating training bottleneck features")
bottleneck_features_trained = base_ResNet50.predict_generator(generator, predict_size_train)

print("saving features")
np.save(train_bottleneck_features_path, bottleneck_features_trained)
bottleneck_features_trained = None


batchsize = 1
print("generating validation features")
generator = datagen.flow_from_directory(valid_path,
                                        target_size=(224,224),
                                        batch_size=batchsize, 
                                        class_mode=None,
                                        shuffle=False)
valid_samples = len(generator.filenames)
predict_size_valid = valid_samples
bottleneck_features_valid = base_ResNet50.predict_generator(generator, predict_size_valid)
np.save(valid_bottleneck_features_path, bottleneck_features_valid)
bottleneck_features_valid = None

print("now generating test features")
generator = datagen.flow_from_directory(test_path,
                                        target_size=(224,224),
                                        batch_size=batchsize, 
                                        class_mode=None,
                                        shuffle=False)
test_samples = len(generator.filenames)
predict_size_test = test_samples
bottleneck_features_test = base_ResNet50.predict_generator(generator, predict_size_test)
np.save(test_bottleneck_features_path, bottleneck_features_test)



