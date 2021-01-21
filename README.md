# Speech Recognition with Spiking Nets

Implementation of the paper "Unsupervised speech recognition through spike-timing-dependent plasticity in a convolutional spiking neural network" by Meng Dong, Xuhui Huang, and Bo Xu, published at PLoS One in November 2018 ([link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204596)).

We implement the convolutional spiking neural network from the paper that is trained to recognize speech uttarances using Spike-Timing Dependent Plasticity (STDP). The network training is fully unsupervised, as the network acts as a feature extractor that is only trained using local STDP. A linear SVM is used to classify the embeddings produced by the network and predict the utterance class from it. The authors use the [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) as well as the [TIDIGITS](https://catalog.ldc.upenn.edu/LDC93S10) dataset to train and test their model. We only focused on TIDIGITS, since documentation for the proper usage of TIMIT was found lacking.

The authors report a classification accuracy of `97.5%` on the test set of the TIDIGITS dataset. In our implementation, we were only able to achieve a `92%` accuracy. For an analysis of the possible reasons for this, as well as detailed documentation of the implementation process, please see [our report]().

## Directory structure

* `vis/`: plots of training progresses as well as network analysis corresponding to the plots found in the paper
* `model/`: model related outputs
  * `logs/`: logs of different training runs including the membrane activations and output potentials of neurons and classification scores during training. Needed for the plots in `vis/`
  * `weights/`: weights of trained models
* `src/`: contains all source code necessary for training, testing and analysing a model
  * `models/`: code that implements model architectures. Contains `speechmodel.py` which implements the network architecture as described in the paper.
  * `utils/`: utility modules:
    - `model/`: modules for training and testing models
    - `data/`: modules for loading the data and transforming them into the MFSC features used in the paper
    - `generic.py`: module for generic helper functions, e.g. status printing for scripts or plotting training progress
  * `run.py`: entry point for interacting with this library. Can be controlled via command line. Run `python src/run.py --help` to see all arguments.
  
