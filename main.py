from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dropout, Flatten, Dense  
from keras.models import Model, Sequential  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
import numpy as np
import os
from glob import glob
from sklearn.datasets import load_files       

test_path = os.path.join("dogImages","test")
training_path = os.path.join("dogImages", "train", "*")


# uncomment the following one if you want to predict using only general weights
# num_classes = 133
# top_model_weights_path = os.path.join('weights','general_breeds.h5') 

# uncomment the following one if you want to predict using only general weights
num_classes = 138
top_model_weights_path = os.path.join('weights','specific_breeds_10epochs.h5') 


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(np.array(data['target']), 138)
    return dog_files, dog_targets

# load test dataset
test_files, test_targets = load_dataset(test_path)
# load list of dog names (this does not correspond to imagenet labels)
dog_names = [item[20:-1] for item in sorted(glob(training_path))]
# num_classes = len(dog_names)

# define ResNet50 model
base_ResNet50 = ResNet50(include_top=False, weights='imagenet')
full_ResNet50 = ResNet50(weights="imagenet")

#shape of the last convolutional layer of resnet50
m_input_shape=(6680,7,7,2048)

top_model = Sequential()  
top_model.add(Flatten(input_shape=m_input_shape[1:]))  
top_model.add(Dense(138, activation='relu'))  
top_model.add(Dropout(0.5))  
top_model.add(Dense(138, activation='softmax'))
top_model.compile(optimizer='rmsprop',  
            loss='categorical_crossentropy', metrics=['accuracy']) 

top_model.load_weights(top_model_weights_path)  


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# Predicts from default resnet50 trained on imagenet1000
def regular_ol_resnet(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    output = full_ResNet50.predict(img)
    return  np.argmax(output)

# Predicts the dog breed / name of an image based off our modified resnet50
def predict_labels(img_path):
    # returns prediction vector for image located at img_path   
    img = preprocess_input(path_to_tensor(img_path))
    bottle_features = base_ResNet50.predict(img)
    output = top_model.predict(bottle_features)
    dog_index = np.argmax(output)
    print(dog_names[dog_index]+ ": score = " + str(output[0][dog_index]))
    return dog_names[int(dog_index)]
    # return dog_names[np.argmax(output), output(np.argmax(output))]

doggy_name = regular_ol_resnet(os.path.join(test_path,"001.Affenpinscher","Affenpinscher_00003.jpg"))
print(doggy_name)
doggy_name = predict_labels(os.path.join(test_path, "001.Affenpinscher","Affenpinscher_00003.jpg"))
#  x = predict_labels(os.path.join(test_path,"001.Affenpinscher","Affenpinscher_00023.jpg"))
print(doggy_name)
doggy_name = predict_labels(os.path.join(test_path,"001.Affenpinscher","Affenpinscher_00023.jpg"))
print(doggy_name)
doggy_name = predict_labels(os.path.join(test_path,'136.Dean','Capture24.png'))
print(doggy_name)
print(predict_labels("test_doge2.jpg"))
# # print(doggy_name)