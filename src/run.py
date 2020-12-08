import argparse

import numpy as np
import timeit

from utils.model.train import Trainer

from models.speechmodel import SpeechModel

def train_model(model, data_path):
    # Initialize the trainer with the path pointing to data on disk
    trainer = Trainer(data_path)
    # Set the trainer for the model
    model.set_trainer(trainer)
    # Fit the model
    model.fit(epochs=2)

def test_model(model, data_path, label_path):
    model.unfreeze()
    pass

def getArgs():
    """ Parse command line arguemnts """

    parser = argparse.ArgumentParser(description="Interact with library.")

    parser.add_argument("--train", 
                        action='store_true',
                        help='Run training')
    parser.add_argument("--test", 
                        action='store_true',
                        help='Run testing')
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
    parser.add_argument("--labels", "--labelpath",
                        type=str,
                        help="Path to labels matching train data")
    parser.add_argument("-v", "--verbose", 
                        dest='verbose', 
                        action='store_true', 
                        help="Verbose mode")

    return parser.parse_args()

if __name__=='__main__':

    CONFIGS = getArgs()

    model = SpeechModel(input_shape = (41,40))

    if CONFIGS.load_weights:
        model.load_weights(path=CONFIGS.load_weights)
    
    if CONFIGS.train:
        # Check if specific path to data is provided
        if CONFIGS.datapath:
            path = CONFIGS.datapath
        # Else use hardcoded default
        else:
            path = "src/utils/data/own_tidigit_train_results.npy"

        train_model(model, path)

    if CONFIGS.freeze:
        # Freeze model
        model.freeze()

    if CONFIGS.dummy_test:
        # Test speed
        model.time_test(n_trials=1, n_timesteps=20)

    if CONFIGS.test:
        # Check whether both paths provided
        if not (CONFIGS.datapath and CONFIGS.labelpath):
            raise ValueError("Both `--datapath` and `--labelpath` need to be provided and correspond to the same data.")
        test_model(model, CONFIGS.datapath, CONFIGS.labelpath)