'''Entrena un MLP en reconocer caracteres
escritos a mano del repositorio MNIST.
'''

from __future__ import print_function
import time
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json

#parametros globales
batch_size = 128
num_classes = 10
epochs = 30

# Separar data en train y val
(x_train, y_train), (x_val, y_val) = mnist.load_data()
data = x_val
x_train = x_train.reshape(60000, 784)
x_val = x_val.reshape(10000, 784)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255 #normalizo la entrada a [0,1]
x_val /= 255
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')


#Convierte los objetivos a categorical
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
#visualizar salida categorica


#Carga la red guardada
json_file = open('mlpmnist.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("mlpmnist.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.3),
              metrics=['accuracy'])
score = loaded_model.evaluate(x_val, y_val, verbose=0)
print('val loss:', score[0])
print('val accuracy:', score[1])
pred = loaded_model.predict(x_val)


for i in range(10):
    print('valor predicho: ',np.argmax(pred[i], axis=0))
    fig, ax = plt.subplots()
    image = data[i]
    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('dropped spines')
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()
    input('press any key...')

