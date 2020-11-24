import numpy as np

class InputLayer():
    def __init__(self, shape):
        self.shape = shape
    
    # TODO Implement conversion of MFSC spectograms to spikes
    def __call__(self, mfsc_input):
        """ Compute time-to-first-spike array from MFSC spectrogram and return
        it as list of 2-D binary spike matrices for discrete timesteps
        """
        
        # Check shape of inputs
        if mfsc_input.shape != self.shape:
            raise ValueError()
    
        pass
        
    def dummy_call(self, n_timesteps):
        """ Return list of `n_timesteps` of `self.shape` sized matrices that
        contain random spikes for neurons """

        return [np.random.choice(a=[1., 0.], size=self.shape, p=[0.01, 0.99])
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
        self.shape = (input_shape[0]-window_size+1, n_featuremaps)
        
        # Since we share weights between several neurons inside of a feature
        # map, the number of the convolutional windows we need to define is
        # reduced; it depends on how many neurons at once share weights
        self.n_windows_per_feature = self.shape[0] // sharing_size
        
        # Initialize the weights from a Gaussian distribution. This includes 
        # the weights for each convolutional window, so it can be seen as a 
        # 2-D matrix with values for each group of neurons that share weights
        # in every featuremap, where each value is the actual convolutional 
        # window, i.e. another 2-D matrix with weights for every input neuron 
        # that is covered by it.
        self.weights = np.random.normal(
            loc=0.0, 
            scale=0.1, 
            size=(self.n_windows_per_feature,
                  self.n_featuremaps,
                  self.window_size,
                  input_shape[1]))
        
        # Membrane voltages, i.e. the internal state of every neuron in 
        # this layer
        self.membrane_voltages = np.zeros(self.shape)
        
        # Parameters for Integrate-and-Fire neurons
        self.v_thresh = 28.0
        self.v_reset = 0.0
        
        # Parameters for STDP (NOTE: not necessarily all are needed)
        self.tauM = 10.0,
        self.timestep = 0.1,
        self.tau_plus = 10.0,
        self.tau_minus = 10.0,
        self.A_plus = 0.005,
        self.A_minus = 1.1*0.005
    
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

        output_spikes = np.zeros(self.shape, dtype=bool)
           
        if self.is_training:
            # If needed, you can loop through the presynaptic (input) spikes
            for row_in in range(self.input_shape[0]):
                for col_in in range(self.input_shape[1]):
                    if spikes[row_in,col_in]:
                        # TODO Update weights

        # Record spikes and update weights
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                
                # Update membrane potential
                self.membrane_voltages[row,col] += np.sum(np.multiply(
                    self.weights[row//self.window_size,col,:,:], 
                    spikes[row:row+self.window_size,:]))
                
                # Post-synaptic spike
                if self.membrane_voltages[row,col] >= self.v_thresh:
                    # Record spike
                    output_spikes[row,col] = True
                    # Reset membrane potential
                    self.membrane_voltages[row,col] = self.v_reset

                    if self.is_training:
                        # TODO Update weights
                    
                    # Lateral inhibition: when one spike in this row has occured
                    # skip all other neurons in this row. NOTE please also 
                    # think about whether this makes sense, this was just my
                    # first idea
                    break
        
        return output_spikes
    
    def reset(self):
        """ Reset all internal states for a new input sample. """
        self.membrane_voltages = np.zeros(self.shape)

# TODO Implement pooling layer
class PoolingLayer():
    """ The pooling layer yet to be implemented. """
    
    def __init__(self):
        pass
    
    def __call__(self, spikes):
        pass
    
class SpeechModel():
    """ The main model that implements the different layers
    in itself and can be trained by running it on images.
    """
    
    def __init__(self, input_shape):
        # Initialize the different layers
        self.input_layer = InputLayer(input_shape)
        self.conv_layer = ConvLayer(input_shape)
        self.pooling_layer = PoolingLayer()
        
    def freeze(self):
        """ Freeze the model weights to disable STDP learning when input is 
        fed to the model. """
        self.conv_layer.is_training = False

    def unfreeze(self):
        """ Unfreeze the model weights to enable STDP learning when input is
        fed to the model. """
        self.conv_layer.is_training = True

    def run_on_image(self, input_mfsc):
        """ Run the SpeechModel on a single MFSC spectrogram. Returns a list 
        of membrane potentials of all neurons in the last layer (PoolingLayer).
        """
        
        # Reset layers
        self.conv_layer.reset()
        
        # Get the spike iterator from the input layer
        spike_frames = self.input_layer.dummy_call(n_timesteps=10)
        
        # Collect membrane potentials from pooling layer
        # that are used as embeddings for t-SNE
        membrane_potentials = np.empty(len(spike_frames))
        
        # Iterate through matrices of binary spikes
        for i, spikes in enumerate(spike_frames):
            # Feed spikes into conv layer 
            spikes = self.conv_layer(spikes)
            
            # NOTE Right now the pooling layer only returns None-types because
            # it's not yet implemented
            membrane_potentials[i] = self.pooling_layer(spikes)
            
        return membrane_potentials

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
            conv_spikes = []
            for spikes in spike_frames:
                conv_spikes.append(self.conv_layer(spikes))

        # Record time for `n_trials` trials
        time = timeit.timeit(run, number=n_trials)
        
        print('Total time: {:.3f}s, Average: {:.3f}s'.format(time,time/n_trials))