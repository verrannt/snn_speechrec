import traceback
import time

import numpy as np

from ..data.mfsc import result_handler
from ..generic import ProgressNotifier

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

        self.train_prog = ProgressNotifier(
            title='Training', total=self.trainsize)
        self.val_prog = ProgressNotifier(
            title='Validating', total=self.valsize, show_bar=False)

    def read_data(self):
        """ Read the data from storage. Get size of the dataset (i.e. number
        of datapoints) and shape of a single datapoint that may be accessed
        from outside. """

        data = result_handler().load_file(self.path)

        # TODO This is only a temporary hard coded fix, because the data are
        # currently provided in a transposed manner. Hence, we need to trans-
        # pose them back
        new_data = np.empty((data.shape[0], data.shape[2], data.shape[1]))
        for i in range(data.shape[0]):
            new_data[i] = data[i].T
        data = new_data

        self.datashape = (data.shape[1], data.shape[2])
        print("Read {} datapoints from storage with shape {}x{}"
            .format(data.shape[0], data.shape[1], data.shape[2]))

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
                
        # Update the training progress notifier
        self.train_prog.update()

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

        # Update the validation progress notifier
        self.val_prog.update()

        return image

    def reset(self):
        """ Reset indexes to zero and reset the progress notifiers. """
        self.trainindex = 0
        self.valindex = 0
        
        self.train_prog.reset()
        self.val_prog.reset()

    def set_model(self, model):
        """ Set the model instance for fitting. Has to be done 
        before calling `self.fit()` """

        if not self.datashape == model.input_layer.input_shape:
            raise ValueError("The data in the trainer has a different shape than what this model was initialized for. Data shape: {}, Model shape: {}"
                .format(self.datashape, model.input_layer.input_shape))

        self.model = model

    def fit(self, epochs):
        """ Fit the model """

        if not self.model:
            raise ValueError("Model is not set. Call `trainer.set_model()` with an appropriate model instance.")
        
        print("Fitting model on {} images".format(self.trainsize))

        # Check if weights are frozen
        if not self.model.conv_layer.is_training:
            self.model.conv_layer.is_training = True
            print("WARNING: model weights were automatically unfrozen")

        # Collect the membrane potentials of the pooling layer for all images
        # in all epochs
        train_potentials = np.empty((
            epochs, 
            self.trainsize, 
            self.model.pooling_layer.output_shape[0], 
            self.model.pooling_layer.output_shape[1]))
        val_potentials = np.empty((
            epochs, 
            self.valsize, 
            self.model.pooling_layer.output_shape[0], 
            self.model.pooling_layer.output_shape[1]))

        # Iterate through all epochs
        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch+1, epochs))
            start_time = time.time()

            # Reset the trainer at the start of each epoch (i.e. index = 0)
            # Also resets progress notifiers
            self.reset()

            # TRAIN on the training data
            for i in range(self.trainsize):
                train_potentials[epoch,i] = self.model(self.next())
                
            print()

            # VALIDATE on the validation data
            self.model.freeze()
            for i in range(self.valsize):
                val_potentials[epoch,i] = self.model(self.valnext())
                # TODO Implement categorization with SVM on the potentials
            self.model.unfreeze()

            # Print elapsed time
            end_time = time.time()
            elapsed_time = end_time-start_time
            print('\nElapsed time {:02}:{:.2f}'
                .format(int(elapsed_time/60), elapsed_time%60))

        print("\nDone.")
        return train_potentials, val_potentials