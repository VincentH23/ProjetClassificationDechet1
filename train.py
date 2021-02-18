from keras.optimizers import SGD
from params import *
from utils import *
from model import *
import matplotlib.pyplot as plt

def training(generatorTrain, generatorVal):
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

    epochs=range(1,NBEPOCH+1)

    loss_train=history.history["loss"]
    metrics_train=history.history["accuracy"]

    loss_test=history.history["val_loss"]
    metrics_test=history.history["val_accuracy"]

    plt.figure()
    plt.title("loss ")
    plt.plot(epochs,loss_train,'r+',label="Training loss")
    plt.plot(epochs,loss_test,'b+',label="Testing loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.show()

    plt.figure()
    plt.title("accuracy ")
    plt.plot(epochs,metrics_train,'r+',label="Training accuracy")
    plt.plot(epochs,metrics_test,'b+',label="Testing accuracy")
    plt.legend()
    plt.xlabel("epochs")
    plt.show()

    print("Nombre d'opérateurs nécessaires par ligne (base de données de validation)")
    print(workforce_needed(generatorVal, model, phase='validation'))
    return model,history

