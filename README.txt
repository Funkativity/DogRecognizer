This is a final project for Computer Vision in our 3D World, as taught Fall of 2018.

It is purely for acedemic purposes at this point.

Our current goal is to implement (using OpenCV or a similar library) a recognizer of dog breeds that identifies dog breeds with greater accuracy than we can.
We will consider our implementation successful if it succeeds at identifying the breed in more of those 100 images than we could.
All related files are in this publicly available github repository https://drive.google.com/drive/u/0/folders/1MWoGmwVRQQ6H4Ei35EVdXQSO92XA3nch.

To recognize dogs:
If you want to use our model trained on general dogs, comment out lines 22 and 23 of main.py, 
and uncomment lines 18 and 19.
Then call predict_image(path-to-image)

If you want to use default resnet trained on imagenet1000, 
just call regular_ol_resnet(path-to-image)


Dependencies:
Keras
tqdm
Tensorflow
Python 3.6.* (3.7 does not support TensorFlow)
Numpy
Glob
sklearn
