import traceback
import time

import numpy as np
from sklearn import svm
from sklearn.utils import shuffle

from ..data.io import load_labels_from_mat, load_data_from_path
from ..generic import ProgressNotifier, DataStream

class Trainer():
    """ 
    Trainer allows to easily train a model on a dataset provided through
    a path variable upon initialization. It reads the data and stores it in
    itself, as to enable easy obtaining of single datapoints with trainer.next()
    """

    def __init__(self, datapath, labelpath, validation_split=0.2):
        """ Initialize the trainer with path to data stored on device """

        # Get the data and labels
        # valdata and vallabels are None if validation_split == 0.0
        traindata, trainlabels, valdata, vallabels = \
            load_data_from_path(datapath, labelpath, validation_split=validation_split)

        self.datashape = traindata.shape[1], traindata.shape[2]

        # Get DataStreams and ProgressNotifiers for training and maybe 
        # validation data
        self.trainstream = DataStream(traindata, trainlabels)
        self.train_prog = ProgressNotifier(
            title='Training', total=self.trainstream.size)
        if valdata is not None:
            self.valstream = DataStream(valdata, vallabels)
            self.val_prog = ProgressNotifier(
                title='Validating', total=self.valstream.size, show_bar=False)
            self.uses_validation = True
        else:
            self.uses_validation = False

    def fit(self, model, epochs):
        """ Fit a model on the internal data """
        
        if self.datashape != model.input_layer.input_shape:
            raise ValueError(
                "The provided model has an input shape different from the "
                "internal data shape. Data shape: {}, Model shape: {}"
                .format(self.datashape, model.input_layer.input_shape))

        print("Fitting model on {} images".format(self.trainstream.size), end='')
        if self.uses_validation:
            print(", validating on {} images".format(self.valstream.size), end='')
        print()

        # Check if weights are frozen
        if not model.conv_layer.is_training:
            model.unfreeze()
            print("WARNING: model weights were automatically unfrozen")

        # Collect the membrane potentials of the pooling layer for all images
        # in all epochs
        train_potentials = np.empty((
            epochs, 
            self.trainstream.size, 
            model.pooling_layer.output_shape[0], 
            model.pooling_layer.output_shape[1]))
        if self.uses_validation:
            val_potentials = np.empty((
                epochs, 
                self.valstream.size, 
                model.pooling_layer.output_shape[0], 
                model.pooling_layer.output_shape[1]))
        else:
            val_potentials = None

        # Iterate through all epochs
        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch+1, epochs))
            start_time = time.time()

            # Reset the trainer at the start of each epoch (i.e. index = 0)
            # Also resets progress notifiers
            self.trainstream.reset()
            self.train_prog.reset()

            # TRAIN on the training data
            for i in range(self.trainstream.size):
                train_potentials[epoch,i] = model(self.trainstream.next())                    
                self.train_prog.update()
            print()
            
            clf = svm.LinearSVC(max_iter=5000)
            clf = clf.fit(
                train_potentials[epoch].reshape(self.trainstream.size,9*50), 
                self.trainstream.labels)
            train_score = clf.score(
                train_potentials[epoch].reshape(self.trainstream.size,9*50), 
                self.trainstream.labels)
                
            print('Training Accuracy: {:.2f}'
                .format(train_score))

            # VALIDATE on the validation data
            if self.uses_validation:

                self.valstream.reset()
                self.val_prog.reset()

                model.freeze()
                for i in range(self.valstream.size):
                    val_potentials[epoch,i] = model(self.valstream.next())
                    self.val_prog.update()
                model.unfreeze()
                print()

                clf = svm.LinearSVC(max_iter=5000)
                clf = clf.fit(
                    val_potentials[epoch].reshape(self.valstream.size,9*50), 
                    self.valstream.labels)
                val_score = clf.score(
                    val_potentials[epoch].reshape(self.valstream.size,9*50), 
                    self.valstream.labels)
                
                print('Validation Accuracy: {:.2f}'
                    .format(val_score))

            # Print elapsed time
            end_time = time.time()
            elapsed_time = end_time-start_time
            print('Elapsed time {:02}:{:02}:{:02}'.format(
                int(elapsed_time/60), 
                int(elapsed_time%60), 
                int(elapsed_time%60%1*100)))

        print('\nFinished training\n')

        return model, train_potentials, val_potentials

