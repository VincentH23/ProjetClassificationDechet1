from keras.optimizers import SGD
from params import *
from utils import *
from model import *

def training(generatorTrain, generatorVal,descenteGrade):
    model=create_model()
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=descenteGrade, 
        metrics=['accuracy'] # métrique à changer

    )

    hist_train=model.fit_generator(
        generatorTrain, epochs=NBEPOCH, callbacks=None,
        validation_data=generatorVal,
        class_weight=None,shuffle= SHUFFLE_DATA
    )

    model.save_weights('./checkpoint')
    return model
B=0
if __name__=='__main__':
    
 