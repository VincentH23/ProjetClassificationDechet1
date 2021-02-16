from keras.optimizers import SGD
from params import *
from utils import *
from model import *
import matplotlib.pyplot as plt

def training(generatorTrain, generatorVal,descenteGrade):
    model=create_model()
    
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=descenteGrade, 
        metrics=['accuracy'] # métrique à changer  workforce_needed_create(1879), my_metric_fn

    )
    # metric_train = []
    # metric_validation = []
    # for epoch in range(NBEPOCH):
    history=model.fit_generator(
    generatorTrain, epochs=NBEPOCH, callbacks=None,
    validation_data=generatorVal,
    class_weight=None,shuffle= SHUFFLE_DATA)
        #list_temp=generatorVal.indexes
        #ytrue= generatorVal.__data_generation(list_temp) #ici
        #ypred=model.predict_generator(generatorVal)
        #print(np.sum(ytrue==ypred)/66)
        # metric_train.append(workforce_needed(generatorTrain, model, phase='train'))
        # metric_validation.append(workforce_needed(generatorVal, model, phase='validation'))
        # plt.plot(metric_train, label='Metric on training')
        # plt.plot(metric_validation, label='Metric on validation')
        # plt.legend()
        # plt.show()
        

        
    model.save_weights('./checkpoint')
    return model,history

