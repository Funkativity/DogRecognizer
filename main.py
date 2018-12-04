from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dropout, Flatten, Dense  
from keras.models import Sequential  
from keras.preprocessing import image     
from keras.utils import np_utils
from tqdm import tqdm
import numpy as np
import os
from glob import glob
from sklearn.datasets import load_files       

test_path = os.path.join("dogImages","test")
training_path = os.path.join("dogImages", "train", "*")
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load test dataset
test_files, test_targets = load_dataset(test_path)
top_model_weights_path = 'dog_recognition_bottleneck_model.h5' 
# load list of dog names (this does not correspond to imagenet labels)
#dog_names = [item[20:-1] for item in sorted(glob(training_path))]
# num_classes = len(dog_names)

# define ResNet50 model
# base_ResNet50 = ResNet50(include_top=False, weights='imagenet')
full_ResNet50 = ResNet50(weights="imagenet")
#our top_model that works on top of resnet50
# top_model = Sequential()  
# top_model.add(Flatten(input_shape=base_ResNet50.output.shape[1:]))  
# top_model.add(Dense(133, activation='relu'))  
# top_model.add(Dropout(0.5))  
# top_model.add(Dense(num_classes, activation='relu'))
# top_model.load_weights(top_model_weights_path)  

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def regular_ol_resnet(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    output = full_ResNet50.predict(img)
    return  np.argmax(output)

def predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    img = img / 255 # not sure if i need to do this, may remove later
    bottle_features = base_ResNet50.predict(img)
    output = top_model.predict(bottle_features)
    return dog_names[np.argmax(output), output(np.argmax(output))]

doggy_name = regular_ol_resnet(os.path.join(test_path,"001.Affenpinscher","Affenpinscher_00003.jpg"))
# doggy_name = predict_labels(test_files + os.path.join("001.Affenpinscher","Affenpinscher_00003.jpg"))

print(doggy_name)
doggy_name = regular_ol_resnet(os.path.join(test_path,"001.Affenpinscher","Affenpinscher_00023.jpg"))
print(doggy_name)
