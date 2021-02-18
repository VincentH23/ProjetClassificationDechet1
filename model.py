from keras.models import Sequential
from params import *
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, DepthwiseConv2D, MaxPool2D
import tensorflow as tf

def create_model2():
    tf.random.set_seed(1234)

    model = Sequential()
    
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

    model.add(Dense(NBNEURONE1,activation="relu"))
    model.add(Dropout(DROPOUT))

    model.add(Dense(NBNEURONE2,activation="relu"))
    model.add(Dropout(DROPOUT))

    model.add(Dense(NBNEURONE3,activation="relu"))
    model.add(Dropout(DROPOUT))

    model.add(Dense(NBNEURONE4,activation="relu"))
    model.add(Dropout(DROPOUT))

    model.add(Dense(NBNEURONE5,activation="softmax"))
    model.summary()

    return model


