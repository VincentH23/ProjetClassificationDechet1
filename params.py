from keras.optimizers import SGD, Adam

# General data parameters

DATASET_PATH = 'dataset-resized'
SHUFFLE_DATA = True

# Data generator parameters
TRAINING_BATCH_SIZE = 128
TRAINING_IMAGE_SIZE = (128, 128)
VALIDATION_BATCH_SIZE = 128
VALIDATION_IMAGE_SIZE = (128, 128)
VALIDATION_DATASET_SIZE = 324
TESTING_BATCH_SIZE = 16
TESTING_IMAGE_SIZE = (128, 128)
TEST_DATASET_SIZE = 324
NUMBER_OF_CHANNELS = 3
TRANSFORM = Tru


# Model parameters
NBNEURONE1 = 128
NBNEURONE2 = 6
taux_apprentissage = 0.0003
NBEPOCH = 100
descenteGrade = Adam(taux_apprentissage)

# Values for metric
WORK_PER_PERSON_FOR_TWENTY_MINS = 80.54
MARGIN = 9