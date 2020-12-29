import traceback
import time

import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import copy

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

        # Determines whether training will be stopped at the end of an episode
        # due to criterion communicated from the model
        self.stop_training = False

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
            print("[!] Warning: model weights were automatically unfrozen")

        # Reset in case it has been set before
        self.stop_training = False

        # Collect the membrane potentials of the pooling layer for all images
        # in all epochs
        train_potentials = np.empty((
            epochs, 
            self.trainstream.size, 
            model.pooling_layer.output_shape[0], 
            model.pooling_layer.output_shape[1]))
        train_scores = []
        if self.uses_validation:
            val_potentials = np.empty((
                epochs, 
                self.valstream.size, 
                model.pooling_layer.output_shape[0], 
                model.pooling_layer.output_shape[1]))
            val_scores = []
        else:
            val_scores = None
            val_potentials = None

        # Keep track of feature map activations to visualize it
        feature_map_activations = []
        if epochs <= 1:
            visualize_freq = 150
        else:
            visualize_freq = 2000

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

                # Stop training with criterion from model
                if model.check_stopping_criterion():
                    self.stop_training = True

                if (epoch * self.trainstream.size + i) % visualize_freq == 0:
                    # Save weights for feature map visualisation
                    feature_map_activations.append([copy.copy(model.conv_layer.weights[4,  0, :, :]),
                                                    copy.copy(model.conv_layer.weights[4, 24, :, :]),
                                                    copy.copy(model.conv_layer.weights[4, 49, :, :])])
            print()

            clf = svm.LinearSVC(max_iter=5000)
            clf = clf.fit(
                train_potentials[epoch].reshape(self.trainstream.size,9*50), 
                self.trainstream.labels)
            train_score = clf.score(
                train_potentials[epoch].reshape(self.trainstream.size,9*50), 
                self.trainstream.labels)
            train_scores.append(train_score)
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

                val_score = clf.score(
                    val_potentials[epoch].reshape(self.valstream.size,9*50), 
                    self.valstream.labels)
                val_scores.append(val_score)
                print('Validation Accuracy: {:.2f}'
                    .format(val_score))

            # Print elapsed time
            end_time = time.time()
            elapsed_time = end_time-start_time
            print('Elapsed time {:02}:{:02}:{:02}'.format(
                int(elapsed_time/60), 
                int(elapsed_time%60), 
                int(elapsed_time%60%1*100)))

            # Stop training
            if self.stop_training:
                print('[!] Stopping criterion was met and training will be '
                'terminated.')
                break

        print('\nFinished training\n')

        self.plot_history(train_scores, val_scores, len(train_scores))
        # Plot some feature maps at different times in training
        if feature_map_activations: # check if not empty
            self.visualize_featuremaps(feature_map_activations, visualize_freq)
        # Plot output of SNN for a sample of each digit
        self.visualize_snn(model)
        
        return model, train_potentials, val_potentials, train_scores, val_scores

    def plot_history(self, train_scores, val_scores, n_epochs):
        fontsize=15
        
        plt.figure(figsize=(20,10))
        plt.plot(range(1, n_epochs+1), train_scores)
        if val_scores:
            plt.plot(range(1, n_epochs+1), val_scores)
        plt.grid(True)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Accuracy', fontsize=fontsize)
        plt.xticks(range(1, n_epochs+1), fontsize=fontsize*0.9)
        plt.yticks(fontsize=fontsize*0.9)
        plt.title('Training History', fontsize=fontsize*1.2)
        plt.show()        

    def visualize_snn(self, model):
        """ Plot the output of the SNN (pooling potentials) for a sample of each digit """
        # Variables to keep track of plotted labels
        labels_used = []
        uniques = set(self.trainstream.labels)

        # Create subplots with general information
        fig, axs = plt.subplots(int(np.ceil(len(uniques) / 2)), 2)
        plt.setp(axs, xticks=[], yticks=[])
        plt.subplots_adjust(hspace=0.5)
        axs[int(np.ceil(len(uniques) / 2) - 1), 0].set_xlabel("Feature maps")
        axs[int(np.ceil(len(uniques) / 2) - 1), 0].set_ylabel("Sections")

        # Make sure that we are not training
        model.freeze()

        done = False
        index = 0
        while not done and index < len(self.trainstream.labels):
            # Get label of current sample
            label = self.trainstream.labels[index]

            # Check if label is already plotted
            if label not in labels_used:
                # Get SNN output of sample
                image = self.trainstream.data[index]
                # Plot SNN output
                axs[int((label - 1) / 2), int((label - 1) % 2)].imshow(model(image), vmin=0, vmax=4)
                axs[int((label - 1) / 2), int((label - 1) % 2)].set_title("Digit " + str(int(label)), size=10)
                # Keep track of plotted labels
                labels_used.append(label)
            index += 1
            # Check if all labels are plotted
            if set(labels_used) == uniques:
                done = True
        # Show final plot
        plt.show()

    def visualize_featuremaps(self, activations, steps):
        """ Plot the feature maps of the SNN (weight of CNN) for three feature maps """
        # Create subplots with general information
        fig, axs = plt.subplots(len(activations), 3)
        plt.setp(axs, xticks=[], yticks=[])
        axs[len(activations) - 1, 0].set_xlabel("Feature map #1")
        axs[len(activations) - 1, 1].set_xlabel("Feature map #25")
        axs[len(activations) - 1, 2].set_xlabel("Feature map #50")
        fig.text(0.05, 0.5, 'Number of training samples', ha='center', va='center', rotation='vertical')

        min_weight = 0
        max_weight = max(1, np.max(np.array(activations)))
        for index, item in enumerate(activations):
            # Set label
            axs[index, 0].set_ylabel(steps * index, rotation='horizontal', labelpad=17)
            # Plot the three feature maps
            axs[index, 0].imshow(item[0], vmin=min_weight, vmax=max_weight)
            axs[index, 1].imshow(item[1], vmin=min_weight, vmax=max_weight)
            axs[index, 2].imshow(item[2], vmin=min_weight, vmax=max_weight)
        # Show final plot
        plt.show()