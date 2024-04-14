# masterthesis


Two folders: scalar-on-function and function-on-function are used to store files related to different tasks. Each of the folders contains the following files:

- `Real` - real datasets and corresponding scripts used for fine-tuning and producing the results.
- `Simulation` - folder `data` that was used to generate the data and the corresponding scripts used to conduct the simulation experiments.
- `Models.py` - all the PyTorch neural network model classes
- `Utils.py` - standard FDA models, PyTorch helper functions, cross-validation, etc.

  Inside the folders `Real` and `Simulation` are the folders with task names e.g. Tecator, Bike Sharing, task 2, etc. that contain the datasets, files facilitating the fine-tuning process via [ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html) library and final scripts that were manually hard-coded with the results obtained during the fine-tuning. This process was needed because of different data formats, number of functional inputs, basis selection and etc. 
