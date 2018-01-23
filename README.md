# Face Segmenter

This is a Fully Convolutional Network built with Keras that is meant to segment faces. It is based of the VGG-16 architecture and implements one skip layer. It uses pretrained weights for the VGG-Face dataset. After about 10 hours of training on an NVIDIA GeForce GTX 1050 over 9 epochs, the model got to about 94% accuracy.

## Weights

Pre-trained weights were downloaded from <http://www.vlfeat.org/matconvnet/pretrained/>.

## Dataset

The dataset I used came from [this repository](https://github.com/arahusky/Tensorflow-Segmentation). It is a modified form of the [lfw part labels](http://vis-www.cs.umass.edu/lfw/part_labels/) dataset.

## Pitfalls

To help others in implementing their own FCN, I've decided to list all of the pitfalls I fell into during this project.

### 1. Class Imbalance

Most of the labels have about a 1/4 ratio of face to background pixels. As a result, my network would classify every pixel as a 0 and remained stuck at about 80% accuracy. To resolve this issue, I used Keras's class_weight parameter for model.fit(). In order to not get the error `ValueError: 'class_weight' must contain all classes in the data.` you need to follow [these instructions](https://stackoverflow.com/questions/48254832/keras-class-weight-in-multi-label-binary-classification) which tell you to set `y_train[:,0] = 0` and `y_train[:,1] = 1`. Also, I used a lambda layer with `K.squeeze` to remove the unnecessary last dimension of the output. If you need class weights for more than two classes then follow [these instructions](https://github.com/keras-team/keras/issues/3653), which describe how to use sample_weight.

### 2. LeakyReLU not ReLU

Over the course of training I found that using ReLU for upsampling would cause the loss to decrease very slowly and often it got stuck around 0.5. Using LeakyReLU produced much better results.

### 3. Poor Optimization

I began by using adam with its defualt settings but I found that this did a poor job of reducing the loss. After doing some research and playing around with optimizers I settled on `SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)` as most effective.
