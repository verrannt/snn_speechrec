import traceback

from ..data.mfsc import result_handler

class Trainer():
    """ 
    Trainer allows to easily train a model on a dataset provided through
    a path variable upon initialization. It reads the data and stores it in
    itself, as to enable easy obtaining of single datapoints with trainer.next()
    """

    def __init__(self, path, validation_split=0.2):
        """ Initialize the trainer with path to data stored on device """
        self.path = path
        self.valsplit = validation_split

        self.trainindex = 0
        self.valindex = 0

        self.read_data()

    def read_data(self):
        """ Read the data from storage. Get size of the dataset (i.e. number
        of datapoints) and shape of a single datapoint that may be accessed
        from outside. """

        data = result_handler().load_file(self.path)
        self.datashape = (self.data.shape[1], self.data.shape[2])

        datasize = data.shape[0]
        self.valsize = int(datasize * self.valsplit)
        self.trainsize = datasize - self.valsize

        val_indices = np.random.choice(
            data.shape[0], self.valsize, replace=False)
        train_indices = np.delete(np.arange(datasize), val_indices)
        
        self.valdata = data[val_indices]
        self.traindata = data[train_indices]

    def next(self):
        """ Get the datapoint for the training data at the current index and 
        increase the index. If the index has reached the end of the dataset, 
        raise an IndexError and notify that index has to be reset. """

        # Fail safe if index has reached end of dataset
        try:
            # Draw the current image
            image = self.traindata[self.trainindex]
        except IndexError:
            raise IndexError("Trainer reached the end of the training data. To use further, call `trainer.reset()` to reset the index to 0.")

        # Increase the index
        self.trainindex += 1

        return image

    def valnext(self):
        """ Get the datapoint for the validation data at the current index and 
        increase the index. If the index has reached the end of the dataset, 
        raise an IndexError and notify that index has to be reset. """

        # Fail safe if index has reached end of dataset
        try:
            # Draw the current image
            image = self.valdata[self.valindex]
        except IndexError:
            raise IndexError("Trainer reached the end of the validation data. To use further, call `trainer.reset()` to reset the index to 0.")

        # Increase the index
        self.valindex += 1

        return image

    def reset(self):
        """ Reset indexes to zero. """
        self.trainindex = 0
        self.valindex = 0