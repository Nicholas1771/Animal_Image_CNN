from keras.preprocessing import image
import numpy as np
from tensorflow.python.keras.models import model_from_json

json_file = open('cnn.json', 'r')
loaded_cnn = json_file.read()
json_file.close()

cnn = model_from_json(loaded_cnn)
cnn.load_weights('cnn_model.h5')
print('Loaded model from disk')


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
