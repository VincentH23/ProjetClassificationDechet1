import numpy as np
import matplotlib.pyplot as plt
from params import *
import tensorflow as tf


def confusion_matrix(x_test, y_test, model):
    
    M = np.zeros((6, 6))
    L = model.predict_classes(x_test)
    for i in range(len(y_test)):
        k = 0
        for j in range (1,6):
            k = j*y_test[i][j]
        M[L[i]][k] += 1
    return M.astype(int)


def show_confusion_matrix(confusion_matrix):
    for i in range(6):
        print(confusion_matrix[i])
    plt.imshow(confusion_matrix)
    plt.colorbar()
    plt.title('Matrice de confusion sur la base de données de test')
    plt.xlabel('Classe réelle')
    plt.ylabel('Classe prévue par le réseau')
    plt.show()


def workforce_needed(generator, model, phase='train'):
    if phase == 'train':
        dataset_size = 2527 - VALIDATION_DATASET_SIZE - TEST_DATASET_SIZE
        batch_size = TRAINING_BATCH_SIZE
    elif phase == 'test':
        dataset_size = TEST_DATASET_SIZE
        batch_size = TESTING_BATCH_SIZE
    elif phase == 'validation':
        dataset_size = VALIDATION_DATASET_SIZE
        batch_size = VALIDATION_BATCH_SIZE
    number_of_batches = dataset_size//batch_size
    y_true = []
    y_pred = []
    for batch_number in range(number_of_batches):
        images, labels = generator.__getitem__(batch_number)
        y_true.append(labels)
        y_pred_to_add = model.predict(images)
        y_pred.append(y_pred_to_add)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nombre_dechets = y_true.shape[0]
    rapport_temps = nombre_dechets / 2527
    work_per_person = WORK_PER_PERSON_FOR_TWENTY_MINS * rapport_temps
    nombre_dechets_mal_tries = [0 for i in range(6)]
    for j in range(nombre_dechets):
        if (y_true[j] != y_pred[j]).all() :
            numero_ligne = 0
            for i in range(6):
                numero_ligne += y_pred[j][i]*i
            nombre_dechets_mal_tries[int(numero_ligne)] += 1
    workforce = [nombre_dechets_mal_tries[i]/work_per_person for i in range(6)]
    return workforce


def workforce_needed_create(dataset_size) :  # return a function
    def  workforce_needed(y_true,y_pred):   #y_true shape =(1,6)  
        work = [0 for i in range(6)]
        # y_pred_decision = tf.argmax(y_pred)
        # y_true_decision = tf.argmax(y_true)
        # if y_true_decision != y_pred_decision :
        #   work[int(y_pred_decision.numpy())] = 1
        return [1, 0]
    return workforce_needed

def my_metric_fn(y_true, y_pred):
  y_true_dec = tf.argmax(y_true,1)
  y_pred_dec = tf.argmax(y_pred,1)
  a = (y_pred_dec != y_true_dec) and (y_pred_dec == 0)
  return a
    
        


