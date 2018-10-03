from utils.model import model
import utils.run_utils as utils
import argparse
import glob

parser = argparse.ArgumentParser(description='Train an FCN')
parser.add_argument('imgs', help='path to training images')
parser.add_argument('labels', help='path to ground truth labels')
parser.add_argument('-e', '--epochs', type=int, help='Number of epochs (10 by default)')

args = parser.parse_args()

img_dir_path = args.imgs
label_dir_path = args.labels
epochs = args.epochs

imgs = glob.glob(img_dir_path + '/' + img_dir_path + '/*')
labels = glob.glob(label_dir_path + '/' + label_dir_path + '/*')

if len(imgs) != len(labels):
    raise ValueError('Error: Different number of images and labels')

if epochs is None:
    epochs = 10

img_path = imgs[0]
label_path = labels[0]

fcn_model = model()

train_generator = utils.image_generator(img_path, label_path, img_dir_path, label_dir_path)

utils.train_model(fcn_model, train_generator, len(imgs), epochs).save('model.h5')