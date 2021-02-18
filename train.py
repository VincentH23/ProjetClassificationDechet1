from keras.optimizers import SGD
from params import *
from utils import *
from model import *
import matplotlib.pyplot as plt

def training(generatorTrain, generatorVal,descenteGrade):
    model=create_model2()
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=descenteGrade, 
        metrics=['accuracy'] # métrique à changer  workforce_needed_create(1879), my_metric_fn

    )

    model.evaluate_generator(
        generatorVal
    )

    history=model.fit_generator(
    generatorTrain, epochs=NBEPOCH, callbacks=None,
    validation_data=generatorVal,
    class_weight=None,shuffle= SHUFFLE_DATA)
             
    model.save('checkpoint')

    print("Nombre d'opérateurs nécessaires par ligne (base de données de validation)")
    print(workforce_needed(generatorVal, model, phase='validation'))
    return model,history

