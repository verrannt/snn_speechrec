import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_labels_from_mat(path, typ:str='train'):
    """ Load class labels from a matlab file provided under `path` and return
    them as 1-D Numpy array of integers corresponding to the digit spoken.
    Currently only works for the TIDIGIT dataset.
    """

    if not path.find('TIDIGIT')>0:
        raise ValueError(
            "Currently, this function works only for the TIDIGITS "
            "dataset, but the provided path did not match `TIDIGIT`.")

    # If path contains 'train'
    if path.find('train')>0:
        which = 'train_labels'
    # If path contains 'test'
    elif path.find('test')>0:
        which = 'test_labels'
    else:
        raise ValueError(
            "Must be train or test labels, but found neither `train` nor "
            "`test` substring in `path`.")

    # Load train or test labels, respectively
    mat_labels = loadmat(path)[which]

    # flatten the matlab labels to 1-D array
    labels = np.empty(mat_labels.shape[0])
    for i in range(labels.shape[0]):
        labels[i] = mat_labels[i][0][0,0]

    return labels

def load_sex_from_df(path):
    """ Just for testing purposes: functionality to load the TIMIT dataset's
     train.csv file and extract speaker sex as labels """

    df = pd.read_csv(path)
    # Remove trailing NaN rows
    df = df.loc[:23099]
    # Get rows corresponding to actual wav files
    df = df.loc[df.filename.str.contains('.wav')]
    # Get speaker ids
    ids = df.speaker_id.to_numpy()
    # Convert to binary
    bids = np.empty(ids.shape[0])
    for i in range(ids.shape[0]):
        if ids[i][0] == 'F':
            bids[i] = 1
        elif ids[i][0] == 'M':
            bids[i] = 0
    return bids