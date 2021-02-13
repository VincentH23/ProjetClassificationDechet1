from keras.models import Sequential
from params import *
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout
import tensorflow as tf

def create_model():
    tf.random.set_seed(1234)

    model = Sequential()
    model.add(Conv2D(32, (3,3),activation="sigmoid", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    model.add(Conv2D(64, (3,3), activation="sigmoid", strides=(1,1), padding = "same"))
    model.add(Conv2D(64, (3,3), activation="sigmoid", strides=(2,2), padding = "same"))
    model.add(Conv2D(64, (3,3), activation="sigmoid", strides=(2,2), padding = "same"))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(NBNEURONE1,activation="sigmoid"))
    model.add(Dense(NBNEURONE2,activation="softmax"))
    model.summary() 

    return model


