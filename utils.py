import numpy as np
import matplotlib.pyplot as plt
from params import *
import tensorflow as tf


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

    #Pour initialiser les listes avec les bonnes dimensions
    images, labels = generator[0]
    y_true = labels
    y_pred = model.predict(images)

    y_true = np.array(y_true)
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.array(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    nombre_dechets_mal_tries = []
    for i in range(6):
        y_ligne=y_pred==i
        dechet_maltriée=y_true[y_ligne]!=i
        nombre_dechets_mal_tries.append(np.sum(dechet_maltriée))
        
    nombre_dechets = y_true.shape[0]
    rapport_temps = nombre_dechets / 2527
    work_per_person = WORK_PER_PERSON_FOR_TWENTY_MINS * rapport_temps
    workforce = [nombre_dechets_mal_tries[i]/work_per_person for i in range(6)]
    return workforce


class Workforce_needed(tf.keras.metrics.Metric):

    def __init__(self,name='workforce',**kwargs):
        super(Workforce_needed, self).__init__(name=name,**kwargs)
        self.counter=self.add_weight(name='counter',initializer='zeros') #nb de dechets visualises au total
        self.nombre_dechets_mal_tries=[self.add_weight(name='nb_mal_triés_ligne_'+str(i), initializer='zeros') for i in range(6)]
        self.workforce=self.add_weight(name='workforce',initializer='zeros')
    
        print('sas30')

    
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.counter.assign_add(tf.reduce_sum(y_true))
        for i in range (6):
            a=tf.argmax(y_true[tf.argmax(y_pred,1)==i],1)!=i
            a=tf.cast(a,tf.float32)
            self.nombre_dechets_mal_tries[i].assign_add(tf.reduce_sum(a))
    

        
    def result(self):
        rapport_temps = self.counter / 2527
        work_per_person = WORK_PER_PERSON_FOR_TWENTY_MINS * rapport_temps
        
        for i in range(6):
            self.workforce.assign_add(tf.math.ceil(self.nombre_dechets_mal_tries[i]/work_per_person))

        return self.workforce

    def reset_states(self):
        for i in range (6):
            self.nombre_dechets_mal_tries[i].assign(0)
        self.workforce.assign(0)
        self.counter.assign(0)
        

