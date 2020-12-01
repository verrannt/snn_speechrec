from src.utils.data.mfsc import *
import numpy as np

def main():
    digit_converter = TIDIGIT_Converter()
    timit_converter = TIMIT_Converter()
    handler = result_handler()

    ######First, TIDIGIT stuff
    #results_own = digit_converter.convert_tidigit_own('TIDIGIT_train.mat', 'train_samples', 20000, 41, 40)
    #results_lib = digit_converter.convert_tidigit_lib('TIDIGIT_train.mat', 'train_samples', 20000, 41, 40)
    #results_own = np.array(results_own)
    #results_lib = np.array(results_lib)

    ######Second, TIMIT stuff
    results_train = timit_converter.convert_timit_own('Spike TIMIT/train', 41, 40)
    print('converted train')
    results_test = timit_converter.convert_timit_own('Spike TIMIT/test', 41,40)
    print('converted test')

    handler.save_file('own_timit_train_results.npy', results_train)
    print('saved train')
    handler.save_file('own_timit_test_results.npy', results_test)
    print('saved test')

    #For saving a numpy array to a file. The .npy extension is necessary
    #handler.save_file('own_tidigit_train_results.npy', results)

    #For loading a file into a numpy array
    #new_results = handler.load_file('own_tidigit_train_results.npy')


if __name__ == '__main__':
    main()
