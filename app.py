from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf


#image
img_width, img_height = 224, 224

#model
top_model_weights_path = 'bottleneck_fc_model.h5'
class_dictionary = np.load('class_indices.npy').item()
print(len(class_dictionary))
num_classes = len(class_dictionary)
print("I'm working...")
model2 = load_model('./cardsfinal2.h5')
graph = tf.get_default_graph()

#app
app = flask.Flask(__name__)


def prepare_image(image, target):

    image = image.resize(target)
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/predict", methods=["POST"])
def predict():
    '''
    Get the image. If image, transform to bytes, predict against bottleneck model
    '''

    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # graph a default workaround for WSGI 
            with graph.as_default():
                image = prepare_image(image, target=(img_width,img_height))
                model = applications.VGG16(include_top=False, weights='imagenet')

                # get the bottleneck prediction from the pre-trained VGG16 model
                bottleneck_prediction = model.predict(image)

                model2 = Sequential()
                model2.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
                model2.add(Dense(256, activation='relu'))
                model2.add(Dropout(0.5))
                model2.add(Dense(num_classes, activation='sigmoid'))

                model2.load_weights(top_model_weights_path)

                # use the bottleneck prediction on the top model to get the final
                # classification
                class_predicted = model2.predict_classes(bottleneck_prediction)

                probabilities = model2.predict_proba(bottleneck_prediction)

                inID = class_predicted[0]

                inv_map = {v: k for k, v in class_dictionary.items()}

                label = inv_map[inID]

                print("Image ID: {}, Label: {}".format(inID, label))

                pred = {"label": label}
                return flask.jsonify(pred)
    

if __name__ == "__main__":
    app.run(debug=True)