from utils import *

def testing(generatorTest,model):

    loss_and_metrics=model.evaluate(generatorTest, batch_size=TEST_DATASET_SIZE)

    print('Erreur de la base de données TEST',loss_and_metrics[0])
    print('Taux de reconnaisance',loss_and_metrics[1])

    print("Nombre d'opérateurs nécessaires par ligne (base de données de test)")
    print(workforce_needed(generatorTest, model, phase='test'))


