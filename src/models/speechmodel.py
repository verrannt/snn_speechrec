import numpy as np

class InputLayer():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    # TODO Implement conversion of MFSC spectograms to spikes
    def __call__(self, mfsc_input):
        """ Compute time-to-first-spike array from MFSC spectrogram and return
        it as list of 2-D binary spike matrices for discrete timesteps
        """
        
        # Check shape of inputs
        if mfsc_input.shape != self.input_shape:
            raise ValueError()
    
        raise NotImplementedError("Calling input layer on MFSC frames is not yet implemented.")
        
    def dummy_call(self, n_timesteps):
        """ Return list of `n_timesteps` of `self.shape` sized matrices that
        contain random spikes for neurons """

        return [np.random.choice(a=[1., 0.], size=self.input_shape, p=[0.20, 0.80])
            for _ in range(n_timesteps)]

# TODO Implement STDP learning in the conv layer. I have tried a little bit
# and you will find some updates using traces but that so far fail because of
# wrong indexing. Calling the conv layer works when `is_training` is False.
# I don't know yet if we can use the traces like so (it's from the NIPS homework)
# because the author's use a slightly different version of STDP. All of the 
# weight updates are below `if self.is_training` statements.
class ConvLayer():
    def __init__(self, 
                 input_shape,
                 n_featuremaps:int=50,
                 window_size:int=6,
                 sharing_size:int=4,
                 is_training:bool=True):
        """
        Spiking convolutional layer with local weight sharing and lateral 
        inhibition. Implements Integrate-and-Fire neurons and updates its 
        weights using STDP if `training` is true.
        
        Parameters
        ----------
        input_shape:
            Shape of the input layer connected to this layer
        n_featuremaps:
            Number of feature maps inside the layer
        window_size:
            Number of rows in the input a single convolutional window covers
        sharing_size:
            How many neighbouring neurons inside a feature map share their 
            weights
        is_training:
            Only if this is True will STDP learning be applied
        """
        
        self.input_shape = input_shape
        self.n_featuremaps = n_featuremaps
        self.window_size = window_size
        self.is_training = is_training
        
        # Compute shape of the convolutional layer depending
        # on the shape of the input layer and size of the 
        # convolutional windows
        self.output_shape = (input_shape[0]-window_size+1, n_featuremaps)
        
        # Since we share weights between several neurons inside of a feature
        # map, the number of the convolutional windows we need to define is
        # reduced; it depends on how many neurons at once share weights
        self.n_windows_per_feature = self.output_shape[0] // sharing_size
        
        # Initialize the weights from a Gaussian distribution. This includes 
        # the weights for each convolutional window, so it can be seen as a 
        # 2-D matrix with values for each group of neurons that share weights
        # in every featuremap, where each value is the actual convolutional 
        # window, i.e. another 2-D matrix with weights for every input neuron 
        # that is covered by it.
        self.weights = np.random.normal(
            loc=0.8,
            scale=0.05,
            size=(self.n_windows_per_feature,
                  self.n_featuremaps,
                  self.window_size,
                  input_shape[1]))
        
        # Membrane voltages, i.e. the internal state of every neuron in 
        # this layer
        self.membrane_voltages = np.zeros(self.output_shape)

        #Record the spiking history of the input neurons, to do STDP
        self.input_spike_history = np.zeros(self.input_shape)
        self.allowed_to_spike = np.ones(self.output_shape)
        self.allowed_to_learn = np.ones(self.output_shape)
        
        # Parameters for Integrate-and-Fire neurons
        self.v_thresh = 28.0
        self.v_reset = 0.0
        
        # Parameters for STDP (NOTE: not necessarily all are needed)
        self.A_plus = 0.004
        self.A_minus = 0.003
        self.delta_weight = 0
    
    def __call__(self, spikes):
        """ Call the convolutional layer on spikes from the input layer. It 
        updates the membrane potentials of each neuron in the conv layer and 
        checks for spikes. Also implements spike-time-dependent plasticity
        for learning if `self.training` is true.

        Parameters
        ----------
        spikes: 2-D binary Numpy array corresponding to spike coordinates
            in the input layer

        Returns
        -------
        output_spikes: 2-D binary Numpy array corresponding to spike
            coordinates of neurons in this convolutional layer
        """
        
        # Check shape of inputs
        if spikes.shape != self.input_shape:
            raise ValueError("Input array of spikes must correspond to the \
                input shape this layer was initialized for. The array you've \
                inputted has shape {} while the required shape is {}".format(
                    spikes.shape, self.input_shape
                ))

        #Update the history of input spikes with the spikes in the current timestep
        self.input_spike_history += spikes

        output_spikes = np.zeros(self.output_shape, dtype=bool)
           
        # Record spikes and update weights
        for row in range(self.output_shape[0]):
            for col in range(self.output_shape[1]):
                
                # Update membrane potential if not inhibited
                self.membrane_voltages[row,col] += self.allowed_to_spike[row,col] * np.sum(np.multiply(
                    self.weights[row//self.window_size,col,:,:], 
                    spikes[row:row+self.window_size,:]))
                
                # Post-synaptic spike
                if self.membrane_voltages[row,col] >= self.v_thresh:
                    # Record spike
                    output_spikes[row,col] = True
                    # Reset membrane potential
                    self.membrane_voltages[row,col] = self.v_reset

                    #Update weights if stpd is allowed for this neuron
                    if self.is_training:
                        # Update for input spike before output spike
                        delta_weights = self.allowed_to_learn[row, col] * self.A_plus * \
                            np.multiply(np.multiply(self.weights[row//self.window_size, col, :, :],
                            (1-self.weights[row//self.window_size,col,:,:])),
                            self.input_spike_history[row:row + self.window_size, :])
                        self.weights[row // self.window_size, col, :, :] += delta_weights
                        # Keep track of the total weight change
                        self.delta_weight += np.sum(abs(delta_weights))

                        # Update for elsewise
                        delta_weights = self.allowed_to_learn[row, col] * -self.A_minus * \
                            np.multiply(np.multiply(self.weights[row // self.window_size, col, :, :],
                            (1 - self.weights[row // self.window_size, col, :, :])),
                            abs(self.input_spike_history[row:row + self.window_size, :]-1))
                        self.weights[row // self.window_size, col, :, :] += delta_weights
                        # Keep track of the total weight change
                        self.delta_weight += np.sum(abs(delta_weights))

                    # Lateral inhibition of neurons in this row
                    self.allowed_to_spike[row, :] = 0
                    # Disallowing row of neurons to learn with STDP
                    self.allowed_to_learn[row, :] = 0
                    # Disallowing neighborhood neurons to learn with STDP
                    self.allowed_to_learn[row-row%self.window_size : row+self.window_size - row%self.window_size, col] = 0
                    # Break for fast inhibition (no unneeded checks done)
                    break

        return output_spikes
    
    def reset(self):
        """ Reset all internal states for a new input sample. """
        self.membrane_voltages = np.zeros(self.output_shape)
        self.input_spike_history = np.zeros(self.input_shape)
        self.allowed_to_spike = np.ones(self.output_shape)
        self.allowed_to_learn = np.ones(self.output_shape)
        self.delta_weight = 0

class PoolingLayer():
    
    def __init__(self, input_shape, pooling_size:int=4):
        """
        Pooling layer that sums spikes of the convolutional layer over 
        timesteps. Doesn't reset potentials until the next sample occurs.

        Parameters
        ----------
        input_shape:
            Shape of the input layer connected to this layer
        pooling_size:
            Size of the pooling window
        """

        self.input_shape = input_shape
        self.pooling_size = pooling_size
        self.output_shape = (int(input_shape[0]/pooling_size), input_shape[1])

        # Membrane voltages, i.e. the internal state of every neuron in
        # this layer
        self.membrane_voltages = np.zeros(self.output_shape)
    
    def __call__(self, spikes):
        """
        Call the pooling layer on spikes from the convolutional layer. It
        computes the membrane potentials of each neuron in the pooling layer with weights of 1.

        Parameters
        ----------
        spikes: 2-D binary Numpy array corresponding to spike coordinates
             in the convolutional layer.

        Returns
        -------
        membrane_voltages: 2-D Numpy array corresponding to summed number of spikes
            in the convolutional layer.
        """
        #Summing over all timesteps
        summed = np.sum(spikes, axis=0)
        #Pooling step:
        for step in range(0, summed.shape[0], self.pooling_size):
            self.membrane_voltages[int(step/self.pooling_size)] = np.sum(summed[step:step+self.pooling_size], axis=0)
        print(self.membrane_voltages)
        return self.membrane_voltages

    def reset(self):
        """ Reset all internal states for a new input sample. """
        self.membrane_voltages = np.zeros(self.output_shape)
    
class SpeechModel():
    """ The main model that implements the different layers
    in itself and can be trained by running it on images.
    """
    
    def __init__(self, input_shape):
        # Initialize the different layers
        self.input_layer = InputLayer(input_shape)
        self.conv_layer = ConvLayer(input_shape)
        self.pooling_layer = PoolingLayer(self.conv_layer.output_shape)
        
    def load_weights(self, path):
        """ Load weights for the model from a numpy array stored on disk.
        Array must be of same shape as weights in convolutional layer. """
        raise NotImplementedError("Loading weights is not yet implemented.")

    def freeze(self):
        """ Freeze the model weights to disable STDP learning when input is 
        fed to the model. """
        self.conv_layer.is_training = False

    def unfreeze(self):
        """ Unfreeze the model weights to enable STDP learning when input is
        fed to the model. """
        self.conv_layer.is_training = True

    def __call__(self, input_mfsc):
        """ Run the SpeechModel on a single MFSC spectrogram frame. Returns a 
        list of membrane potentials of all neurons in the last layer 
        (PoolingLayer). Whether the model is learning while performing on this
        image depends on whether the weights are frozen or not.
        """
        
        # Reset layers
        self.conv_layer.reset()
        self.pooling_layer.reset()
        
        # Get the spike representations from the input layer
        spike_frames = self.input_layer(input_mfsc)
        
        # Iterate through matrices of binary spikes
        conv_spikes = []
        for spikes in spike_frames:
            conv_spikes.append(self.conv_layer(spikes))
            
        pooling_potentials = self.pooling_layer(conv_spikes)
        return pooling_potentials

    def time_test(self, n_trials, n_timesteps):
        """ Test execution time of network using dummy calls on the input 
        layer """

        print('Running {} tests with {} timesteps each'.format(
            n_trials, n_timesteps))

        import timeit

        spike_frames = self.input_layer.dummy_call(n_timesteps=n_timesteps)
        
        # A single run on the network
        def run():
            self.conv_layer.reset()
            self.pooling_layer.reset()
            conv_spikes = []
            for spikes in spike_frames:
                conv_spikes.append(self.conv_layer(spikes))
            if self.conv_layer.is_training and self.conv_layer.delta_weight < 0.01:
                print('Training stopped because weight changes became insufficient')
                self.conv_layer.is_training = False
            pooling_potentials = self.pooling_layer(conv_spikes)
            #Classifier should work on these pooling_spikes
        # Record time for `n_trials` trials
        time = timeit.timeit(run, number=n_trials)
        
        print('Total time: {:.3f}s, Average: {:.3f}s'.format(time,time/n_trials))
