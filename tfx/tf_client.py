import requests
import random

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import json


print(tf.__version__)

# change into: tf.compat.v1.   in 2.0
tf.logging.set_verbosity(tf.logging.ERROR)


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images/255.0
test_images = test_images/255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape:{}, of {}'.format(
    train_images.shape, train_images.dtype))
print('test_iamges.shape: {}, of {}'.format(
    test_images.shape, test_images.dtype))


def show(idx, title):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28, 28))
    plt.axis('off')
    plt.title('\n\n{}'.format(title), fontdict={'size': 16})


rando = random.randint(0, len(test_images)-1)
show(rando, 'An Example Image:{}'.format(class_names[test_labels[rando]]))


data = json.dumps({"signature_name": "serving_default",
                   "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))

headers = {"content-type": "application/json"}
json_response = requests.post(
    'http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

show(0, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[0])], test_labels[0], class_names[np.argmax(predictions[0])], test_labels[0]))
