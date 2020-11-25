import argparse

import numpy as np
import timeit

from models.speechmodel import SpeechModel

def getArgs():
    """ Parse command line arguemnts """

    parser = argparse.ArgumentParser(description="Interact with library.")

    parser.add_argument("--train", 
                        action='store_true',
                        help='Run training')
    parser.add_argument("--dummy_test", 
                        action='store_true',
                        help='Dummy test network and record time')
    parser.add_argument("--freeze",
                        action='store_true',
                        help='Freeze model parameters (i.e. no STDP)')
    parser.add_argument("-l", "--load_weights", 
                        type=str,
                        help="Path to weights stored as numpy array file.")
    parser.add_argument("-d", "--datapath", 
                        type=str,
                        help="Path to train set")
    parser.add_argument("-v", "--verbose", 
                        dest='verbose', 
                        action='store_true', 
                        help="Verbose mode")

    return parser.parse_args()

if __name__=='__main__':

    CONFIGS = getArgs()

    # Init model
    model = SpeechModel(input_shape = (41,40))

    if CONFIGS.load_weights:
        model.load_weights(path=CONFIGS.load_weights)

    if CONFIGS.train:
        raise NotImplementedError('Training not implemented')

    if CONFIGS.freeze:
        # Freeze model
        model.freeze()

    if CONFIGS.dummy_test:
        # Test speed
        model.time_test(n_trials=10, n_timesteps=20)