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

    for epoch in range(NBEPOCH):
        model.fit_generator(
        generatorTrain, epochs=1, callbacks=None,
        validation_data=generatorVal,
        class_weight=None,shuffle= SHUFFLE_DATA)
        list_temp=generatorVal.indexes
        ytrue= generatorVal.__data_generation(list_temp) #ici
        ypred=model.predict_generator(generatorVal)
        print(np.sum(ytrue==ypred)/66)
        
        

    model.save_weights('./checkpoint')
    return model


