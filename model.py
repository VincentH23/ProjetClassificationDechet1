from keras.models import Sequential
from params import *
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, DepthwiseConv2D, MaxPool2D
import tensorflow as tf

def create_model():
    tf.random.set_seed(1234)

    model = Sequential()
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D

    model.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    model.add(Conv2D(64, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    model.add(Conv2D(128, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    model.add(Conv2D(256, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    model.add(Conv2D(256, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    model.add(Conv2D(256, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(NBNEURONE1,activation="relu"))
    model.add(Dense(NBNEURONE2,activation="softmax"))
    model.summary()

    #model.add(Conv2D(32, (3,3),activation="sigmoid", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    #model.add(Conv2D(64, (3,3), activation="sigmoid", strides=(1,1), padding = "same"))
    #model.add(Conv2D(64, (3,3), activation="sigmoid", strides=(2,2), padding = "same"))
    #model.add(Conv2D(64, (3,3), activation="sigmoid", strides=(2,2), padding = "same"))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(NBNEURONE1,activation="sigmoid"))
    #model.add(Dense(NBNEURONE2,activation="softmax"))
    #model.summary() 

    return model


