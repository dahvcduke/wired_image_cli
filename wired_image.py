import os
import emoji
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import sys
import click
import numpy as np

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import math
import cv2
import tensorflow as tf
from pyfiglet import Figlet
from tqdm import tqdm
import _pickle as pickle


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")
click.echo("")

f = Figlet(font='slant')
click.echo(f.renderText('WIRED! Image CLI'))

click.echo('Wired! Image CLI Is A Command Line Interface for Image Analysis. This CLI allows users to develop image categorization and object detection interfaces from a folder of images. It was developed throught the Wired! Lab at Duke University {}'.format(emoji.emojize(':zap:', use_aliases=True)))


click.echo("")
click.echo("")
click.echo("")
click.echo("")

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
#train_data_dir = 'data4/train'
#validation_data_dir = 'data4/validation'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16
graph = tf.get_default_graph()

model = keras.applications.VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

#########################################################
######## CLOSEST IMAGE ########################################
#########################################################

def get_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def get_closest_images(query_image_idx, num_results=5):
    distances = [ distance.euclidean(pca_features[query_image_idx], feat) for feat in pca_features ]
    idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]
    return idx_closest

def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

#########################################################
######## FINETUNE KERAS ########################################
#########################################################

def prepare_image(image, target):

    image = image.resize(target)
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    return image

#########################################################
######## BOTTLENECK ########################################
#########################################################

def save_bottlebeck_features(train_data_dir, validation_data_dir, epochs, batch_size):
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=int(batch_size),
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / int(batch_size)))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=int(batch_size),
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / int(batch_size)))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)

#########################################################
######## TRAIN ########################################
#########################################################

def train_top_model(train_data_dir, validation_data_dir, epochs, batch_size):
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=int(batch_size),
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    train_labels = generator_top.classes
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=int(batch_size),
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')

    history = model.fit(train_data, train_labels,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        validation_data=(validation_data, validation_labels),verbose=2)

    model.save_weights(top_model_weights_path)

    (eval_loss) = model.evaluate(
        validation_data, validation_labels, batch_size=int(batch_size), verbose=1)

    #print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    # summarize history for accuracy
    model.save('cardsfinal2.h5')
    print("saved model!")


#########################################################
######## PREDICT ########################################
#########################################################

def predict(input_image):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    image_path = input_image

    orig = cv2.imread(image_path)

    click.secho("......processing image.......", fg='blue')
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]


    # get the prediction label
    click.echo("")
    click.echo("")
    click.secho("##########################################", fg='blue')
    click.echo("")
    click.secho("I'm predicting....", fg='blue')
    click.secho("a {}!".format(label), fg='green')
    click.echo("")
    click.secho("##########################################", fg='blue')
    click.echo("")
    click.echo("")


class Document(object):
    def __init__(self, home=None, debug=False):
        self.home = os.path.abspath(home or '.')
        self.debug = debug


@click.group()
@click.option('--document')
@click.pass_context
def cli(ctx, document):
    ctx.obj = Document(document)

@cli.command()
@click.option('--train_data_dir', prompt='the training directory', help='training directory ex ./my_dataset/train')
@click.option('--validation_data_dir', prompt='the test directory', help='validation directory ex ./my_dataset/validation')
@click.option('--epochs', prompt='number of epochs', help='number of epochs to train model ex. 50')
@click.option('--batch_size', prompt='batch size', help='bactch size for training ex. 16')
def train_model(train_data_dir, validation_data_dir, epochs, batch_size):
    save_bottlebeck_features(train_data_dir, validation_data_dir, epochs, batch_size)
    train_top_model(train_data_dir, validation_data_dir, epochs, batch_size)

@cli.command()
@click.option('--input_image', prompt='You Message', help='input image, ex. my_image.png')
def predict_class(input_image):
    predict(input_image)

@cli.command()
@click.option('--input_image', prompt="Input Image Folder", help='input image folder, ex. ./my_image_folder')
@click.option('--max_num', prompt='Max number of Images to Train', help='max number of images ex. 500')
def closest_train(input_image, max_num):
    max_num_images = int(max_num)
    images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_image) for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    if max_num_images < len(images):
        images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]

    features = []
    for image_path in tqdm(images):
        img, x = get_image(image_path);
        feat = feat_extractor.predict(x)[0]
        features.append(feat)

    features = np.array(features)
    pca = PCA(n_components=300)
    pca.fit(features)
    pca_features = pca.transform(features)

    print("keeping %d images to analyze" % len(images))
    pickle.dump([images, pca_features], open('./features_images.p', 'wb'))

@cli.command()
@click.option('--input_image', prompt="Input Image Number", help='input image folder, ex. ./my_image_folder')
def closest_predict(input_image):
    images, pca_features = pickle.load(open('./features_images.p', 'rb'))

    query_image_idx = input_image
    idx_closest = get_closest_images(query_image_idx)
    query_image = get_concatenated_images([query_image_idx], 300)
    results_image = get_concatenated_images(idx_closest, 200)
    click.echo(results_image)

if __name__ == '__main__':
    cli()
    