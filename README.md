# ML-Boilerplate

Boilerplate repository for machine learning projects. This assumes a primarily scripting-based setup, in which most interaction is implemented using a central file `src/run.py` that loads/processes data and loads/trains/tests models defined in separate modules.

## Directory structure

* `vis/`: all types of visualisations, like plots and images of model architectures
* `model/`: model related outputs, like training and testing logs, model weights and results
* `src/`: contains all source code in different subdirectories:
  * `models/`: code that implements model architectures
  * `notebooks/`: Jupyter Notebooks for quick testing and visualization that are not part of the main workflow
  * `utils/`: utility modules:
    - `model/`: modules for loading, training and testing your models
    - `data/`: modules for loading and preprocessing/augmenting your data
    - `generic.py`: module for generic helper functions, e.g. status printing for scripts
  * `run.py`: main file in which to set up the logical flow of the project and run different interactions, i.e. call the functions defined in the different modules above
  
**Note:** there is no central `data/` directory as data might be placed outside of this repository. If not, I recommend to place it at the root. 
