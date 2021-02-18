import argparse
import os
from data import *
from test import testing
from train import training
from params import *

parser = argparse.ArgumentParser(description='')

parser.add_argument('--phase', dest='phase', default='train', help="'train' pour entraîner ou 'test' pour tester")

args = parser.parse_args()


def main():

    trainGenerator, valGenerator, testGenerator = create_generators()

    if args.phase == 'train':
        model, history = training(trainGenerator, valGenerator)

    elif args.phase == 'test':
        
        file_path= "./checkpoint"
        model=tf.keras.models.load_model(file_path)

        test(testGenerator, model)

    else:
        print("/!\ Unknown phase : type 'train' or 'test'")
        exit(0)


if __name__ == '__main__':
    main()