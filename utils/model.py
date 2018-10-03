from keras.models import Sequential, Model
from keras.layers import Lambda, LeakyReLU, Reshape, Add, Cropping2D, Conv2DTranspose, Permute, Conv2D, MaxPooling2D, Activation
from keras import backend as K

def model(train_vgg=False):
    """Defines a VGG-16 based FCN with one skip connection

    Args: train_vgg (bool, optional): False by default. Set to True if you would like to train the
                                      VGG-16 portion along with the upsampling portion.

    Returns: a Keras funtional model
    """
        #First half of model is the VGG-16 Network with the last layer removed

    #Input is 224x224 RGB images
    K.set_image_data_format( 'channels_last' )

    model = Sequential()     
    model.add(Permute((1,2,3), input_shape=(224, 224, 3)))
        
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', name="conv1_1", trainable=train_vgg))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', name="conv1_2", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', name="conv2_1", trainable=train_vgg))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', name="conv2_2", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_1", trainable=train_vgg))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_2", trainable=train_vgg))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_3", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_1", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_2", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_3", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_1", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_2", trainable=train_vgg))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_3", trainable=train_vgg))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
            
    model.add(Conv2D(4096, kernel_size=(7,7), padding='same', activation='relu', name='fc6', trainable=train_vgg))
    model.add(Conv2D(4096, kernel_size=(1,1), padding='same', activation='relu', name='fc7', trainable=train_vgg))

        #Second half of model upsamples output to a 224x224 grayscale image

    #reduce the number of channels to 1
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same', activation='relu', name='score_fr'))

    #First Convolution Transpose
    model.add(Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='valid', name='upsample1'))
    model.add(LeakyReLU())

    #trim output for skip connection
    model.add(Cropping2D(cropping=((0, 2), (0, 2))))

    #Reduce number of channels of layer-14 to 1 so it can be added to the output of previous layer
    score14 = LeakyReLU()(Conv2D(1, kernel_size=(1,1), padding='same', name='score14')(model.layers[14].output))

    #Skip connection
    skip1 = Add()([score14, model.layers[-1].output])

    #Perform two more convolutional transposes
    up2 = LeakyReLU()(Conv2DTranspose(1,kernel_size=(8,8), strides=(4,4), padding='valid', name='upsample2')(skip1))
    crop_margin = Cropping2D(cropping=((2, 2), (2, 2)))(up2)

    up2 = LeakyReLU()(Conv2DTranspose(1,kernel_size=(8,8), strides=(4,4), padding='valid', name='upsample3')(crop_margin))
    crop_margin2 = Cropping2D(cropping=((2, 2), (2, 2)))(up2)

    #Flatten the output and remove the last dimensionm
    model = Model(model.input, Lambda(lambda y: K.squeeze(y, -1))(Activation('sigmoid')(Reshape((224*224, 1))(crop_margin2))))

    return model

