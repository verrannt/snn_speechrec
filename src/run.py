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

    if CONFIGS.train:
        print('Training not implemented')

    if CONFIGS.dummy_test:
        # Init model
        model = SpeechModel(input_shape = (41,40))

        # Freeze model because STDP is not correctly implemented
        model.freeze()

        # Test speed
        model.time_test(n_trials=10, n_timesteps=20)