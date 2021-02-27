import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# data preprocessing

# preprocessing training set

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

print(training_set.class_indices)

# preprocessing test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = train_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# building CNN

# initialize the CNN
cnn = tf.keras.models.Sequential()

# block 1
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    kernel_initializer='he_uniform',
    activation='relu',
    padding='same',
    input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# block 2
cnn.add(tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    kernel_initializer='he_uniform',
    activation='relu',
    padding='same'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# flattening layer
cnn.add(tf.keras.layers.Flatten())

# add full connection layers
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=4, activation='softmax'))

# compile the cnn
opt = SGD(lr=0.001, momentum=0.09)
cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the cnn
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

cnn_model = cnn.to_json()
with open('cnn.json', 'w') as json_file:
    json_file.write(cnn_model)

cnn.save_weights('cnn_model.h5')
print('Saved model to disk')


# make predictions

def make_prediction(img, actual='unknown', name=''):
    result = cnn.predict(img / 255)
    print('Prediction for ' + name + '(' + actual + '): ['
          + np.str_(round(result[0][0] * 100)) + '% cat]['
          + np.str_(round(result[0][1] * 100)) + '% chicken]['
          + np.str_(round(result[0][2] * 100)) + '% dog]['
          + np.str_(round(result[0][3] * 100)) + '% horse]')
    return


def prepare_img(img_name):
    """Prepare image for prediction"""
    img = image.load_img('dataset/single_prediction/' + img_name, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


make_prediction(prepare_img('vio1.jpeg'), name='Vio', actual='dog')
make_prediction(prepare_img('cat1.jpg'), actual='cat')
make_prediction(prepare_img('chicken1.jpeg'), actual='chicken')
make_prediction(prepare_img('horse1.jpeg'), actual='horse')
make_prediction(prepare_img('dog1.jpg'), actual='dog')
make_prediction(prepare_img('cat2.jpg'), actual='cat')
make_prediction(prepare_img('chicken2.jpeg'), actual='chicken')
make_prediction(prepare_img('horse2.jpeg'), actual='horse')
make_prediction(prepare_img('dog2.jpg'), actual='dog')
make_prediction(prepare_img('dogecoin.png'), actual='dog')
make_prediction(prepare_img('vio2.jpg'), name='Vio', actual='dog')
make_prediction(prepare_img('vio3.jpg'), name='Vio', actual='dog')
make_prediction(prepare_img('vio4.jpg'), name='Vio', actual='dog')
make_prediction(prepare_img('vio5.jpg'), name='Vio', actual='dog')
