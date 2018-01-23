from keras.models import Sequential, Model
from keras.layers import Lambda, LeakyReLU, Reshape, Add, Cropping2D, Conv2DTranspose, Permute, Conv2D, MaxPooling2D, Flatten, Activation
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tables

f = tables.open_file('data.h5', mode='r')

x_train = f.root.train_i[()]
y_train = f.root.train_t[()]
x_val = f.root.val_i[()]
y_val = f.root.val_t[()]
x_test = f.root.test_i[()]
y_test = f.root.test_t[()]

f.close()

y_train = y_train.reshape(y_train.shape[0], 224*224)
y_val = y_val.reshape(y_val.shape[0], 224*224)
y_test = y_test.reshape(y_test.shape[0], 224*224)

y_train[:,0] = 0
y_train[:,1] = 1

K.set_image_data_format( 'channels_last' )
        
model = Sequential()
        
model.add(Permute((1,2,3), input_shape=(224, 224, 3)))
        
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', name="conv1_1", trainable=False))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', name="conv1_2", trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))
        
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', name="conv2_1", trainable=False))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu', name="conv2_2", trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))
        
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_1", trainable=False))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_2", trainable=False))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu', name="conv3_3", trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))
        
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_1", trainable=False))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_2", trainable=False))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv4_3", trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))
        
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_1", trainable=False))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_2", trainable=False))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu', name="conv5_3", trainable=False))
model.add(MaxPooling2D((2,2), strides=(2,2)))
        
model.add(Conv2D(4096, kernel_size=(7,7), padding='same', activation='relu', name='fc6', trainable=False))
model.add(Conv2D(4096, kernel_size=(1,1), padding='same', activation='relu', name='fc7', trainable=False))

model.add(Conv2D(1, kernel_size=(1, 1), padding='same', kernel_initializer='TruncatedNormal', name='score_fr'))
model.add(LeakyReLU())
convsize = model.layers[-1].output_shape[2]
model.add(Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='valid',  kernel_initializer='TruncatedNormal', name='score2'))
model.add(LeakyReLU())
output_size = (convsize - 1) * 2 + 4
extra_margin = output_size - convsize * 2
c = ((0, extra_margin), (0, extra_margin))
model.add(Cropping2D(cropping=c))

convsize = model.layers[-1].output_shape[2]
output_size = (convsize-1)*16+32
score_pool4 = LeakyReLU()(Conv2D(1, kernel_size=(1,1), padding='same', kernel_initializer='TruncatedNormal', name='score_pool4')(model.layers[14].output))
added = Add()([score_pool4, model.layers[-1].output])
up = LeakyReLU()(Conv2DTranspose(1,kernel_size=(32,32), strides=(16,16), padding='valid', kernel_initializer='TruncatedNormal', name='upsample')(added))
extra_margin = int((output_size - convsize*16) / 2 )
crop_margin = Cropping2D(cropping=((extra_margin, extra_margin), (extra_margin, extra_margin)))
model = Model(model.input, Lambda(lambda y: K.squeeze(y, -1))(Activation('sigmoid')(Reshape((224*224, 1))(crop_margin(up)))))       


from scipy.io import loadmat
data = loadmat('vgg_face_matconvnet/vgg_face_matconvnet/data/vgg_face.mat', 
        matlab_compatible=False,
        struct_as_record=False)
    
net = data['net'][0,0]
l = net.layers
kNames = [layer.name for layer in model.layers]   
a = (0,1,2,3)
for i in range(l.shape[1]):
    mName = l[0,i][0,0].name[0]
    if mName in kNames:
        kIndex = kNames.index(mName)
        weights = l[0,i][0,0].weights[0,0]
        weights = weights.transpose(a)
        bias = l[0,i][0,0].weights[0,1]
        model.layers[kIndex].set_weights([weights, bias[:,0]])

        
from keras.optimizers import SGD
sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val) batch_size=1, epochs=9, verbose=1, class_weight={0: .2, 1: .8})
model.evaluate(x=x_test, y=y_test, verbose=1)
model.save('face_model.h5')