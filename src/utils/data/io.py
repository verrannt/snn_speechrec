import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.utils import shuffle as sklearn_shuffle

from ..data.mfsc import result_handler

def load_data_from_path(datapath:str, 
                        labelpath:str,
                        validation_split:float=0.0,
                        shuffle:bool=True,
                        random_seed=123):
        
        data = result_handler().load_file(datapath)
        labels = load_labels_from_mat(labelpath)
        
        if shuffle:
            data, labels = sklearn_shuffle(data, labels, 
                random_state=random_seed)

        assert data.shape[0] == labels.shape[0], \
            "Data and labels do not fit in shape"

        # TODO This is only a temporary hard coded fix, because the data are
        # currently provided in a transposed manner. Hence, we need to trans-
        # pose them back
        new_data = np.empty((data.shape[0], data.shape[2], data.shape[1]))
        for i in range(data.shape[0]):
            new_data[i] = data[i].T
        data = new_data

        # Get data shape
        datashape = (data.shape[1], data.shape[2])
        print("Read {} datapoints from storage with shape {}x{}"
            .format(data.shape[0], data.shape[1], data.shape[2]))

        # Get size of data and compute size of validation and training set 
        # from provided validation split
        datasize = data.shape[0]

        if validation_split > 0.0:
            valsize = int(datasize * validation_split)
            trainsize = datasize - valsize

            # Randomly choose indices for validation and training set 
            # corresponding to previously defined sizes
            np.random.seed(random_seed)
            val_indices = np.random.choice(
                data.shape[0], valsize, replace=False)
            train_indices = np.delete(np.arange(datasize), val_indices)
        
            # Get the data and labels
            valdata = data[val_indices]
            traindata = data[train_indices]
            vallabels = labels[val_indices]
            trainlabels = labels[train_indices]

            return traindata, trainlabels, valdata, vallabels
        
        else:
            return data, labels, None, None

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