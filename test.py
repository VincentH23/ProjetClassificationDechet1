from utils import *

def test(generatorTest,model):

    loss_and_metrics=model.evaluate(generatorTest, batch_size=TEST_DATASET_SIZE)
    
    print("Nombre d'opérateurs nécessaires par ligne (base de données de test)")
    print(workforce_needed(generatorTest, model, phase='test'))


