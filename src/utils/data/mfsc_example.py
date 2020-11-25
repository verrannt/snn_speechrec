from src.utils.data.mfsc import *
import numpy as np

def main():
    converter = TIDIGIT_Converter()
    handler = result_handler()
    results = converter.convert_tidigit_own('TIDIGIT_test.mat', 'test_samples', 20000, 41, 40)
    results = np.array(results)

    #For saving a numpy array to a file. The .npy extension is necessary
    handler.save_file('own_tidigit_test_results.npy', results)

    #For loading a file into a numpy array
    #new_results = handler.load_file('own_tidigit_train_results.npy')


    #NOTE: THE FOLLOWING LINES ARE FOR TESTING OF RANGES. SHOULD NOT BE REMOVED YET.
    #maxValue = np.amax(results)
    #minValue = np.amin(results)
    #averValue = np.average(results)
    #mat = scipy.io.loadmat('TIDIGIT_train.mat')
    #printIt(results, mat, 0)

if __name__ == '__main__':
    main()
