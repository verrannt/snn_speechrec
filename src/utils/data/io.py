import numpy as np
import pandas as pd

def load_labels(path):
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