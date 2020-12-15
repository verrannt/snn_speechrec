import argparse

import numpy as np
import timeit

from models.speechmodel import SpeechModel
from utils.model.train import Trainer
from utils.model.test import Tester

def getArgs():
    """ Parse command line arguemnts """

    parser = argparse.ArgumentParser(description="Interact with library.")

    parser.add_argument("--train", 
                        type=int,
                        nargs='?',
                        const=1,
                        help='Run training. If followed by integer, train the '
                        'model for this amount of epochs. Otherwise, trains '
                        'for one epoch only.')
    parser.add_argument("--test", 
                        action='store_true',
                        help='Run testing. Usually, you would provide saved '
                        'weights to be loaded with the --load_weights flag, '
                        'or run testing directly after training.')
    parser.add_argument("--dummy_test", 
                        action='store_true',
                        help='Dummy test network and record time.')
    parser.add_argument("--freeze",
                        action='store_true',
                        help='Freeze model parameters (i.e. no STDP).')
    parser.add_argument("--load", 
                        type=str,
                        help="Load weights from file stored as numpy array. "
                        "Only provide name of the file (without .npy ending), "
                        "path is hardcoded.")
    parser.add_argument("--save", 
                        type=str,
                        help="Save weights and potentials stored as numpy "
                        "array files. Only provide info about the run, e.g. "
                        "`run1_epoch1`, the rest is hardcoded.")
    parser.add_argument("-d", "--train_data", 
                        type=str,
                        help="Path to train set. If none is provided, will "
                        "default to path hardcoded in this script.")
    parser.add_argument("-l", "--train_labels",
                        type=str,
                        help="Path to labels matching train data. If none is "
                        "provided, will default to path hardcoded in this "
                        "script.")
    parser.add_argument("--test_data", 
                        type=str,
                        help="Path to test set. If none is provided, will "
                        "default to path hardcoded in this script.")
    parser.add_argument("--test_labels",
                        type=str,
                        help="Path to labels matching test data. If none is "
                        "provided, will default to path hardcoded in this "
                        "script.")
    parser.add_argument("-v", "--verbose", 
                        dest='verbose', 
                        action='store_true', 
                        help="Verbose mode. Not supported right now.")

    return parser.parse_args()

if __name__=='__main__':

    CONFIGS = getArgs()

    model = SpeechModel(input_shape = (41,40), n_time_options=30)

    weights_path = 'models/weights/'

    if CONFIGS.load:

        model.load_weights(path='{}weights_{}.npy'.format(
            weights_path, CONFIGS.load))
        print('Loaded model weights')
    
    if CONFIGS.train:

        # Check if specific path to data is provided
        if CONFIGS.train_data and CONFIGS.train_labels:
            datapath = CONFIGS.train_data
            labelpath = CONFIGS.train_labels

        # Else use hardcoded default
        else:
            datapath = "src/utils/data/own_tidigit_train_results.npy"
            labelpath = "data/Spike TIDIGITS/TIDIGIT_train.mat"
            print('No train data was provided, defaulting to the following:\n'
                  ' datapath:  {}\n'
                  ' labelpath: {}'.format(datapath, labelpath))

        # Create trainer for this data
        trainer = Trainer(datapath, labelpath, validation_split=0.2)
        
        # Fit the model on the data
        model, train_potentials, val_potentials = \
            trainer.fit(model, epochs=CONFIGS.train)

    if CONFIGS.save:

        model.save_weights(path='{}weights_{}.npy'.format(
            weights_path, CONFIGS.save))
        print('Saved model weights')
        
        # Save the membrane potentials
        tp_filename = 'models/logs/train_potentials_{}.npy'.format(CONFIGS.save)
        vp_filename = 'models/logs/val_potentials_{}.npy'.format(CONFIGS.save)
        with open(tp_filename, 'wb') as f:
            np.save(f, train_potentials)
        with open(vp_filename, 'wb') as f:
            np.save(f, train_potentials)
        print('Saved potentials')

        print()

    if CONFIGS.freeze:
        # Freeze model
        model.freeze()

    if CONFIGS.dummy_test:
        # Test speed
        model.time_test(n_trials=1, n_timesteps=20)

    if CONFIGS.test:
        # Check whether both paths are provided
        if CONFIGS.test_data and CONFIGS.test_labels:
            test_datapath = CONFIGS.test_data
            test_labelpath = CONFIGS.test_labels
        else:
            test_datapath = "src/utils/data/own_tidigit_test_results.npy"
            test_labelpath = "data/Spike TIDIGITS/TIDIGIT_test.mat"
            print('No test data was provided, defaulting to the following:\n'
                  ' datapath:  {}\n'
                  ' labelpath: {}'.format(test_datapath, test_labelpath))

        tester = Tester(test_datapath, test_labelpath)
        tester.evaluate(model)