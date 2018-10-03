# Image Segmentation on Faces

This is a Fully Convolutional Network built with Keras that is meant to segment faces. It is based of the VGG-16 architecture and implements one skip layer. It uses pretrained weights trained on the VGG-Face dataset.

The model was trained on an Nvidia GTX 1050. After training for about 4 hours on a batch size of 2, over 9 epochs, the model acheived 92% accuracy. 

To see a general diagram of the model architecture, refer to FCN.jpg.

- [Weights](#weights)
- [Dataset](#dataset)
- [Train Your Own Model](#train-your-own-model)
    - [Dependencies](#dependencies)
    - [Getting Started](#getting-started)
- [Pitfalls](#pitfalls)
    - [1. Class Imbalance](#1-class-imbalance)
    - [2. LeakyReLU not ReLU](#2-leakyrelu-not-relu)
    - [3. Poor Optimization](#3-poor-optimization)


## Weights

Pre-trained weights were downloaded from <http://www.vlfeat.org/matconvnet/pretrained/>.

## Dataset

I used the [lfw part labels](http://vis-www.cs.umass.edu/lfw/part_labels/) dataset to train on. 

For the training set I combined the recommended images for the train and test sets. For the test set I used the recommend images from the validation set. I did not have a validation set.

## Train Your Own Model

### Dependencies

- [Numpy](https://www.scipy.org/scipylib/download.html)
- [Keras](https://keras.io/#installation)
- [Tensorflow](https://www.tensorflow.org/install/)
- [Pillow](https://pillow.readthedocs.io/en/5.2.x/installation.html)

These can all be eaisly installed using [pip](https://pypi.org/project/pip/) or [anaconda](https://www.anaconda.com/).

### Getting Started

Clone or download the repository.

Place all of your training images in one folder and your ground truth images in another folder. Nest each of these folders in two more seperate folders so that your folders look something like this.

```
.
├──images
|   ├──images
|
├──labels
    ├──labels
```

Then run train_model.py with first the path to the folder containing your images folder and second to the folder containing your labels folder.

```python
python train_model.py imgs labels
```
By default the model will train for 10 epochs. If you want to train for a different number, use the `-e` option.

```python
python train_model.py imgs labels -e 5
```

## Pitfalls

To help others in implementing their own FCN, I've decided to list all of the pitfalls I ran into during this project.

### 1. Class Imbalance

Most of the labels have about a 1/4 ratio of face to background pixels. As a result, my network would classify every pixel as a 0, causing it to remain stuck at about 80% accuracy. To resolve this issue, I used Keras's `class_weight` parameter for `model.fit()`. 

In order to not get the error `ValueError: 'class_weight' must contain all classes in the data.`, you need to follow [these instructions](https://stackoverflow.com/questions/48254832/keras-class-weight-in-multi-label-binary-classification) which tell you to set `y_train[:,0] = 0` and `y_train[:,1] = 1`. 

If you need class weights for 3+ dimensional output tensors then follow [these instructions](https://github.com/keras-team/keras/issues/3653), which describe how to use `sample_weight`.

To remove the unnecessary last dimension of the output I used a lambda layer with `K.squeeze`. 

### 2. LeakyReLU not ReLU

Over the course of training I found that using ReLU for upsampling would cause the loss to decrease very slowly and often it got stuck around 0.5. Using LeakyReLU produced much better results.

### 3. Poor Optimization

I began by using adam with its defualt settings but I found that this did a poor job of reducing the loss. After doing some research and playing around with optimizers I settled on `SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)` as most effective.
