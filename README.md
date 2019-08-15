# Image Recognition

This python code will help you Image Classifier as a beginner and also train your images and to make predection. The credits for this code go to pranoyr. I've merely fleshed out to get people started.

# Pre-requistes

TensorFlow:
```
$ sudo pip install tensorflow
```
Keras:
```
$ pip install keras
```
matplotlib:
```
$ pip install matplotlib
```
After you downaloded the code unzip the file and create new 2 folders in the repository one is "dataset_image" and predicit.

# Dataset

open the folder which you have created "Dataset_image" then create a sub-folder on the images that you wanted to recognize by the machine. Eg. If you want to make a classify Cat and Dog. Create a folder named Cat and put the images of Cat from the Internet, then create a folder named the dog and put the images of Dog there. By using ImageNet you will get 1000s of images of different varieties of things.

# Training

Ones you prepared your dataset it is the time to train. Open terminal or command prompt in the folder the run the "file train.py" by `$ pyhton train.py`

# Prediction

Ones you completed the training then come back to the same the repo and put the image you wanted to predict. If you wanted to predict cat (Take your phone and click an image of a cat then share it to the computer then put the file in predict folder. Open terminal in the folder and type * `$ python predict.py`

# Useful Links

TensorFlow: https://tensorflow.org

Keras: https://keras.io

# Common errors

[errorno 29] Not a directory: 'Dataset_image/.DS_Store' Go to load_data.py file the find label=label[1:] and make it as comment #label=label[1:]. If you are using Ubuntu you should comment it and if you are using a mac you should not comment it. If you find any more bug report me in rtfbuse@gmail.com
