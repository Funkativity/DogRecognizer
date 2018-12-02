This is a final project for Computer Vision in our 3D World, as taught Fall of 2018.

It is purely for acedemic purposes at this point.

Our current goal is to implement (using OpenCV or a similar library) a recognizer of dog breeds that identifies dog breeds with greater accuracy than we can.
To test this, we will train on a dataset, and then attempt to catagorize 100 images ourselves, and then run the recognizer on those 100 images.
We will consider our implementation successful if it succeeds at identifying the breed in more of those 100 images than we could.
Our images (for both training and testing) will be pulled from http://vision.stanford.edu/aditya86/ImageNetDogs/?fbclid=IwAR2HdY5Ox4YT-KImp67K89v8uAV8rgUZgl4RnUdIHzN1EcEKM5PV9sOb7-0


Dependencies:
Keras
tqdm
Tensorflow
Python 3.6.* (3.7 does not support TensorFlow)
Numpy
Glob
sklearn
