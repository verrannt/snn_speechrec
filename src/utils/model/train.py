import traceback

from ..data.mfsc import result_handler

class Trainer():
    """ 
    Trainer allows to easily train a model on a dataset provided through
    a path variable upon initialization. It reads the data and stores it in
    itself, as to enable easy obtaining of single datapoints with trainer.next()
    """

    def __init__(self, path):
        """ Initialize the trainer with path to data stored on device """
        self.path = path
        self.index = 0
        self.read_data()

    def read_data(self):
        """ Read the data from storage. Get size of the dataset (i.e. number
        of datapoints) and shape of a single datapoint that may be accessed
        from outside. """
        self.data = result_handler().load_file(self.path)
        self.datasize = self.data.shape[0]
        self.datashape = (self.data.shape[1], self.data.shape[2])

    def next(self):
        """ Get the datapoint at the current index and increase the index.
        If the index has reached the end of the dataset, raise an IndexError 
        and notify that index has to be reset. """

        # Fail safe if index has reached end of dataset
        try:
            # Draw the current image
            image = self.data[self.index]
        except IndexError:
            raise IndexError("Trainer reached the end of the dataset. To use further, call `trainer.reset()` to reset the index to 0.")

        # Increase the index
        self.index += 1

        return image

    def reset(self):
        """ Reset index to zero. """
        self.index = 0