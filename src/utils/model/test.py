import traceback
import time

import numpy as np
from sklearn import svm
from sklearn.utils import shuffle

from ..data.io import load_labels_from_mat, load_data_from_path
from ..generic import ProgressNotifier, DataStream


class Tester():

    def __init__(self, datapath, labelpath):
        """ Load test data from provided paths and create a DataStream for this
        data to use in `self.evaluate()` """

        data, labels, _, _ = load_data_from_path(datapath, labelpath)

        self.datashape = data.shape[1], data.shape[2]

        self.stream = DataStream(data, labels)
        self.prog = ProgressNotifier(
            title='Testing', total=self.stream.size)

    def evaluate(self, model):

        if self.datashape != model.input_layer.input_shape:
            raise ValueError(
                "The provided model has an input shape different from the "
                "internal data shape. Data shape: {}, Model shape: {}"
                .format(self.datashape, model.input_layer.input_shape))

        print("Testing model on {} images"
            .format(self.stream.size))

        # Check if weights are frozen
        if model.conv_layer.is_training:
            model.freeze()
            print("WARNING: model weights were automatically frozen")

        potentials = np.empty((
            self.stream.size, 
            model.pooling_layer.output_shape[0], 
            model.pooling_layer.output_shape[1]))
        
        # TEST on the testing data
        for i in range(self.stream.size):
            potentials[i] = model(self.stream.next())                    
            self.prog.update()
        print()

        # Reset stream and prog notifier
        self.stream.reset()
        self.prog.reset()
        
        # Fit classifier on the potentials
        clf = svm.LinearSVC(max_iter=5000)
        clf = clf.fit(
            potentials.reshape(self.stream.size,9*50), 
            self.stream.labels)
        score = clf.score(
            potentials.reshape(self.stream.size,9*50), 
            self.stream.labels)
            
        print('Testing Accuracy: {:.2f}'
            .format(score))

        return potentials

