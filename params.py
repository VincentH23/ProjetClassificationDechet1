# General data parameters
from keras.optimizers import SGD

DATASET_PATH = 'dataset-resized'
SHUFFLE_DATA = True

# Data generator parameters
TRAINING_BATCH_SIZE = 16
TRAINING_IMAGE_SIZE = (128, 128)
VALIDATION_BATCH_SIZE = 16
VALIDATION_IMAGE_SIZE = (128, 128)
VALIDATION_DATASET_SIZE = 324
TESTING_BATCH_SIZE = 16
TESTING_IMAGE_SIZE = (128, 128)
TEST_DATASET_SIZE = 324
NUMBER_OF_CHANNELS = 3


# Model parameters
NBNEURONE1 = 64
NBNEURONE2 = 6
taux_apprentissage = 0.001
NBEPOCH = 5
descenteGrade = SGD(taux_apprentissage)

# Values for metric
WORK_PER_PERSON_FOR_TWENTY_MINS = 80.54
MARGIN = 9