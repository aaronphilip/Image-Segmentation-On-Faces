from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import Image
import numpy as np

def image_generator(img_path, mask_path, img_dir_path, mask_dir_path):
    """Function to create a image generator for fit_generator

    https://keras.io/preprocessing/image/#imagedatagenerator-class

    Args:
        img_path (str): path to a single face image
        mask_path (str): path to a single ground truth image
        img_dir_path (str): path to a directory contaning training images
        mask_dir_path (str): path to a directory containing ground truth training labels
    
    Returns:
        a generator for fit_generator
    """
    data_gen_args = dict(rotation_range=20.,
                     zoom_range=0.2,
                    horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed=1
    image_datagen.fit(np.expand_dims(np.asarray(Image.open(img_path)), axis=0), augment=True, seed=seed)
    mask_datagen.fit(np.expand_dims(np.expand_dims(np.asarray(Image.open(mask_path)), axis=0), axis = 4), augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        img_dir_path,
        class_mode=None,
        seed=1,
        batch_size=1,
        target_size=(224,224)
        )

    mask_generator = mask_datagen.flow_from_directory(
        mask_dir_path,
        class_mode=None,
        seed=1,color_mode='grayscale',
        batch_size=1,
        target_size=(224,224))

    def gt_gen():
        while True:
            y = (mask_generator.next() / 255.).astype(np.float32)
            y = y.reshape((y.shape[0], 224 * 224))
            y[:,0] = 0.
            y[:,1] = 1.
            yield y

    train_generator = zip(image_generator, gt_gen())

    return train_generator


def train_model(model, train_generator, num_imgs, num_epochs=10):
    """function to train a keras fcn model

    Args:
        model (Keras model): an fcn model
        train_generator (generator): a generator to pass to fit_generator
        num_imgs (int): number of images to train in one epoch
        num_epochs (int): number of epochs to train on

    Returns:
        trained FCN
    """
    sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=num_imgs, epochs=num_epochs, class_weight={1:.75, 0:.25}, verbose=1)

    return model