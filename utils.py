import numpy as np
import matplotlib.pyplot as plt
from params import *


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


def workforce_needed(y_true, y_pred):
    nombre_dechets = y_true.shape[0]
    rapport_temps = nombre_dechets / 2527
    work_per_person = WORK_PER_PERSON_FOR_TWENTY_MINS * rapport_temps
    nombre_dechets_mal_tries = [0 for i in range(6)]
    for j in range(nombre_dechets):
        if y_true[j] != y_pred[j]:
            numero_ligne = 0
            for i in range(6):
                numero_ligne += y_pred[j][i]*i
            nombre_dechets_mal_tries[numero_ligne] += 1
    workforce = [nombre_dechets_mal_tries[i]/work_per_person for i in range(6)]
    return workforce
    
        


