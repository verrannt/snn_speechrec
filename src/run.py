import argparse

import numpy as np
import pickle
import timeit

from models.speechmodel import SpeechModel
from utils.model.train import Trainer
from utils.model.test import Tester
from utils.data.io import load_data_from_path

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
    parser.add_argument("--plot_history",
                        type=str,
                        help="Plot training history from disk by providing "
                        "name of run, similar to --load and --save.")
    parser.add_argument("--plot_features",
    parser.add_argument("--plot_featuremaps",
                        action='store_true',
                        help="Plot progess of selected feature maps from "
                        "disk. Provide name of the run using the --load flag.")
    parser.add_argument("--plot_outputs",
                        action='store_true',
                        help="Plot the output of the SNN (pooling potentials) "
                        "for a sample of each digit, given a trained model "
                        "using the --load flag.")
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
        model, train_potentials, val_potentials, train_scores, val_scores, \
            activations, freq = trainer.fit(model, epochs=CONFIGS.train)

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

        # Save training scores as a dictionary
        history = dict()
        history['train_acc'] = train_scores
        if val_scores:
            history['val_acc'] = val_scores
        history_filename = 'models/logs/train_history_{}.npy'.format(CONFIGS.save)
        with open(history_filename, 'wb') as f:
            pickle.dump(history, f)
        print('Saved history')

        # Save feature map activations as dictionary
        act_dict = dict()
        for index, item in enumerate(activations):
            act_dict[index*freq] = item
        activations_filename = 'models/logs/activations_{}.npy'.format(CONFIGS.save)
        with open(activations_filename, 'wb') as f:
            pickle.dump(act_dict, f)
        print('Saved history')

        print()

    if CONFIGS.plot_history:
        history_filename = 'models/logs/train_history_{}.npy'\
            .format(CONFIGS.plot_history)
        with open(history_filename, 'rb') as f:
            history = pickle.load(f)
        train_scores = history['train_acc']
        try:
            val_scores = history['val_acc']
        except KeyError:
            val_scores = None
        Trainer.plot_history(None, train_scores, val_scores, len(train_scores))

    if CONFIGS.plot_features:
    if CONFIGS.plot_featuremaps:
        if not CONFIGS.load:
            raise ValueError('Using this flag requires providing the name of '
                'the run using the --load flag.')

        activations_filename = 'models/logs/activations_{}.npy'\
            .format(CONFIGS.load)
        with open(activations_filename, 'rb') as f:
            act_dict = pickle.load(f)
        visualize_freq = list(act_dict.keys())[1]
        activations = [act_dict[k] for k in act_dict.keys()]

        Trainer.visualize_featuremaps(None, activations, visualize_freq)

    if CONFIGS.plot_outputs:
        if not CONFIGS.load:
            raise ValueError('Using this flag requires loading model weights '
                'using the --load flag.')

        # NOTE This is a hardcoded default to the TIDIGIT data
        datapath = "src/utils/data/own_tidigit_train_results.npy"
        labelpath = "data/Spike TIDIGITS/TIDIGIT_train.mat"
        Trainer(datapath, labelpath, validation_split=0.2).visualize_snn(model)
        
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

        # NOTE The below is quite a hacky and ugly solution to get the training
        # labels and potentials into the `Tester`. There are likely nicer 
        # solutions, but this works for now.

        # Load training potentials
        if CONFIGS.load:
            run_name = CONFIGS.load
        elif CONFIGS.save:
            run_name = CONFIGS.save
        else:
            raise ValueError(
                'Script was called with the --test flag, yet no save nor load '
                'name was provided.')
        run_name = CONFIGS.load if CONFIGS.load else CONFIGS.save
        run_name = 'models/logs/train_potentials_{}.npy'.format(run_name)
        with open(run_name, 'rb') as f:
            train_potentials = np.load(f)[-1]
        # Load training labels
        train_path = test_datapath.replace('test','train')
        train_labelpath = test_labelpath.replace('test','train')
        _,train_labels,_,_ = load_data_from_path(train_path, train_labelpath, 0.2)

        # Run testing
        tester = Tester(test_datapath, test_labelpath)
        tester.evaluate(model, train_potentials, train_labels)

