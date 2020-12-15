from src.utils.data.mfsc import *
import numpy as np

def main():
    #digit_converter = TIDIGIT_Converter()
    timit_converter = TIMIT_Converter()
    handler = result_handler()

    ######First, TIDIGIT stuff
    #results_own = digit_converter.convert_tidigit_own('TIDIGIT_train.mat', 'train_samples', 20000, 41, 40)
    #results_lib_train_digit = digit_converter.convert_tidigit_lib('TIDIGIT_train.mat', 'train_samples', 20000, 41, 40)
    #results_lib_test_digit = digit_converter.convert_tidigit_lib('TIDIGIT_test.mat', 'test_samples', 20000, 41, 40)
    #results_own = np.array(results_own)
    #results_lib_train_digit = np.array(results_lib_train_digit)
    #results_lib_test_digit = np.array(results_lib_test_digit)

    ######Second, TIMIT stuff
    results_lib_train_timit = timit_converter.convert_timit_lib('Spike TIMIT/train', 41, 40)
    print('converted train')
    results_lib_test_timit = timit_converter.convert_timit_lib('Spike TIMIT/test', 41,40)
    print('converted test')

    results_lib_train_timit = np.array(results_lib_train_timit)
    results_lib_test_timit = np.array(results_lib_test_timit)

    handler.save_file('lib_timit_train_results.npy', results_lib_train_timit)
    print('saved train')
    handler.save_file('lib_timit_test_results.npy', results_lib_test_timit)
    print('saved test')

    #For saving a numpy array to a file. The .npy extension is necessary
    #handler.save_file('own_tidigit_train_results.npy', results)

    #For loading a file into a numpy array
    #new_results = handler.load_file('own_tidigit_train_results.npy')
    newMax = np.amax(results_lib_train_timit)
    print(newMax)
    newMax2 = np.amax(results_lib_test_timit)
    print(newMax2)
    print(results_lib_train_timit[0].shape)
    print(results_lib_test_timit[0].shape)


if __name__ == '__main__':
    main()
