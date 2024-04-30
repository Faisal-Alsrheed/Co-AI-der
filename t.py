# Purpose: This script is used to run tests on a Python file and generate a results file with the test results and the contents of the Python file and the test file.
# Author: Faisal Alsrheed
# v2

# Python
import os
import subprocess
from datetime import datetime

# Get the Python file path from the user
python_file = input("Enter the full path to the Python file: ")

# Derive the test file name and output file name from the Python file name
test_file = os.path.splitext(python_file)[0] + "_test.py"

# Get current date and time
now = datetime.now()

# Format as a string
now_str = now.strftime("%Y%m%d_%H%M%S")

# Append to filename
output_file = os.path.splitext(python_file)[0] + f"_results_{now_str}.txt"

backend_dict = {
    "1": "jax",
    "2": "torch",
    "3": "tensorflow",
    "4": "mlx",
    "all": ["jax", "torch", "tensorflow", "mlx"]
}

output_choice = input("Do you want the output to include 'all' (the code, test, and results), or 'test only'? Enter 'all' or 'test only': ")

project_description = """
I am working on a project called "Keras 3.0," a multi-backend implementation of the Keras API designed to support TensorFlow, JAX, and PyTorch.

Keras 3 is a multi-backend deep learning framework that supports JAX, TensorFlow, and PyTorch. It allows effortless building and training of models for computer vision, natural language processing, audio processing, timeseries forecasting, recommender systems, and more.

**Install Keras:**
```bash
pip install keras --upgrade
```
**Install backend package(s):**  
To use Keras, you should also install your backend of choice: `tensorflow`, `jax`, or `torch`. 


**Configuring Your Backend:**  
You can export the environment variable `KERAS_BACKEND` or edit your local config file at `~/.keras/keras.json` to configure your backend. Available backend options are: "tensorflow", "jax", "torch". Example:

```bash
export KERAS_BACKEND="jax"
```
In Colab, you can do:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```
**Note:** The backend must be configured before importing keras, and the backend cannot be changed after the package has been imported.


Run your high-level Keras workflows on top of any framework—benefiting at will from the advantages of each framework, e.g., the scalability and performance of JAX or the production ecosystem options of TensorFlow.

Write custom components (e.g., layers, models, metrics) that you can use in low-level workflows in any framework. You can take a Keras model and train it in a training loop written from scratch in native TF, JAX, or PyTorch. You can take a Keras model and use it as part of a PyTorch-native Module or as part of a JAX-native model function.

Make your ML code future-proof by avoiding framework lock-in. As a PyTorch user: gain access to the power and usability of Keras. As a JAX user: access a fully-featured, battle-tested, well-documented modeling and training library.






The `keras-master` folder contains several subdirectories, which are structured as follows:

- **`.devcontainer`**: An empty directory, likely used for defining development container configurations (e.g., for use with VSCode Remote Containers).

- **`.github`**: Contains GitHub-specific configurations, including:
  - **`workflows`**: Contains GitHub Actions workflow definitions. Subdirectories under workflows are:
    - **`config`**: Contains configuration for different frameworks (JAX, NumPy, TensorFlow, Torch).
    - **`scripts`**: Likely contains scripts used by GitHub Actions workflows.

- **`.kokoro`**: Used for continuous integration configurations, specifically for Google's internal CI system. It includes:
  - **`github`**: Contains configurations for GitHub, further subdivided by platform (Ubuntu) and configurations for GPU with frameworks like JAX, TensorFlow, and Torch.

- **`benchmarks`**: Contains benchmarking scripts or configurations for Keras, divided into:
  - **`layer_benchmark`**, **`model_benchmark`**, and **`torch_ctl_benchmark`**: Each directory likely contains scripts for benchmarking layers, models, and Torch CTL respectively.

- **`examples`**: An empty directory, probably intended for Keras usage examples.

- **`guides`**: An empty directory, possibly for user guides or tutorials.

- **`integration_tests`**: Contains integration tests for the Keras library.

- **`keras`**: The core Keras library directory, containing subdirectories for different components of Keras:
  - **`activations`**, **`applications`**, **`backend`**, **`callbacks`**, **`constraints`**, **`datasets`**, **`distribution`**, **`dtype_policies`**, **`export`**, **`initializers`**, **`layers`**, **`legacy`**, **`losses`**, **`metrics`**, **`models`**, **`ops`**, **`optimizers`**, **`quantizers`**, **`random`**, **`regularizers`**, **`saving`**, **`testing`**, **`trainers`**, **`utils`**: Each directory is dedicated to a specific area of functionality within Keras, such as layers, models, optimizers, and utilities for neural network training and inference.

- **`shell`**: An empty directory, possibly intended for shell scripts related to Keras.

Each of these directories and subdirectories organizes the code, tests, examples, and documentation related to different aspects of the Keras library. The structure reflects a modular approach to organizing a large codebase, with clear separation between different types of content (code, tests, documentation, etc.) and different areas of functionality within the library.


The `keras` folder within the `keras-master` directory contains subdirectories that organize the various components of the Keras library. Here's a brief overview of each subfolder and its intended purpose:

### Core Components and Functionality

- **`activations`**: Contains activation functions, which are mathematical operations that determine the output of a neural network node given an input or set of inputs.

- **`applications`**: Hosts pre-trained models and architectures for deep learning applications. These are well-known models that can be used for transfer learning, such as VGG, ResNet, and Inception.

- **`backend`**: Provides a unified interface to different backend engines (like TensorFlow, Theano, or Microsoft Cognitive Toolkit) and includes subdirectories for specific backend implementations (`common`, `jax`, `numpy`, `tensorflow`, `tests`, `torch`). This design allows Keras to run on top of different computational engines, making the library more flexible.

  - **`common`**: Likely contains shared backend utilities or interfaces.
  - **`jax`**: Contains the JAX backend implementation.
  - **`numpy`**: For the NumPy backend, enabling operations with NumPy arrays.
  - **`tensorflow`**: TensorFlow backend implementation, connecting Keras to TensorFlow's functionalities.
  - **`tests`**: Contains tests specific to backend operations.
  - **`torch`**: PyTorch backend implementation, with a further `optimizers` subfolder possibly for PyTorch-specific optimizers.

- **`callbacks`**: Includes callback functions, which are a set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states and statistics of the model during training.

- **`constraints`**: Houses constraints that can be applied to the network parameters, such as weight constraints for regularization.

- **`datasets`**: Provides easy access to standard datasets for training models, like MNIST, CIFAR-10, etc.

- **`distribution`**: Likely relates to utilities or mechanisms for distributed training, enabling models to be trained on multiple machines or GPUs.

- **`dtype_policies`**: Contains code related to data type policies, which manage how layers compute their output data types based on input types.

- **`export`**: Deals with exporting Keras models, potentially to different formats or platforms.

- **`initializers`**: Includes initializers that set the initial random weights of Keras layers.

### Layers and Models

- **`layers`**: A crucial part of the Keras library, this directory contains various types of neural network layers (the building blocks of models), including:
  
  - **`activations`**, **`attention`**, **`convolutional`**, **`core`**, **`merging`**, **`normalization`**, **`pooling`**, **`preprocessing`**, **`regularization`**, **`reshaping`**, **`rnn`**: Each subdirectory focuses on a specific category of layers, such as convolutional layers for processing images, recurrent layers for sequential data, and so on.

- **`legacy`**: Contains code that has been deprecated or is kept for backward compatibility. This ensures that older models or scripts remain functional.

  - **`preprocessing`** and **`saving`**: Specific functionalities that have newer alternatives but are kept for compatibility.

- **`losses`**: Defines loss functions, which measure how well the model's predictions match the target data.

- **`metrics`**: Contains metrics to evaluate the models, such as accuracy, precision, and recall.

- **`models`**: Provides the API for defining and training models, including the Sequential and Functional APIs.

### Optimization and Regularization

- **`optimizers`**: Includes optimization algorithms, like SGD, Adam, etc., which are strategies used to update weights in the network during training. The `schedules` subfolder likely contains learning rate schedules.

- **`quantizers`**: Pertains to quantization strategies, potentially for reducing model size or for use in environments with limited precision.

- **`random`**: May contain utilities for generating random numbers or patterns, important for initialization and stochastic processes in training.

- **`regularizers`**: Houses regularization methods, which help prevent overfitting by penalizing large weights.

### Saving, Testing, and Utilities

- **`saving`**: Deals with saving and loading Keras models, including the architecture and weights.

- **`testing`**: Contains testing utilities and frameworks to ensure the library's components work as expected.

- **`trainers`**: Includes the `data_adapters` subfolder, likely related to mechanisms for feeding data into the model for training.

- **`utils`**: A collection of utility functions and classes that provide general utility functionalities, like data preprocessing, model visualization, and more.


We are currently working to add another backend called MLX. MLX is a NumPy-like array framework designed for efficient and flexible machine learning on Apple silicon, developed by Apple's machine learning research team.

The Python API closely follows NumPy, with a few exceptions. MLX also features a fully-featured C++ API, which closely follows the Python API.

The main differences between MLX and NumPy are:
- **Composable Function Transformations:** MLX has composable function transformations for automatic differentiation, automatic vectorization, and computation graph optimization.
- **Lazy Computation:** Computations in MLX are lazy. Arrays are only materialized when needed.
- **Multi-Device:** Operations can run on any of the supported devices (CPU, GPU, …)

The design of MLX is inspired by frameworks like PyTorch, JAX, and ArrayFire. A notable difference from these frameworks and MLX is the unified memory model. Arrays in MLX live in shared memory, and operations on MLX arrays can be performed on any supported device type without data copies. The currently supported device types are CPU and GPU.
"""

with open(output_file, 'w') as f:
    f.write(project_description)

    if output_choice.lower() == 'all':
        # 1st: Write Python file contents
        with open(python_file, 'r') as pf:
            f.write(f"\n\n-- Python File ({python_file}) Contents --\n")
            f.write(pf.read())

        # 2nd: Write Test file contents
        with open(test_file, 'r') as tf:
            f.write(f"\n\n-- Test File ({test_file}) Contents --\n")
            f.write(tf.read())

# 3rd: Run the tests and redirect the output to the output file
backend_input = input("Enter the backend you want to use (1 for jax, 2 for torch, 3 for tensorflow, 4 for mlx, 'all' for all): ")

if backend_input.lower() == 'all':
    for backend in backend_dict['all']:
        with open(output_file, 'a') as f:
            f.write(f"\n\n-- Running tests with backend: {backend} --\n")
        subprocess.run(f"KERAS_BACKEND={backend} pytest {test_file} >> {output_file} 2>&1", shell=True)
else:
    backend = backend_dict.get(backend_input)
    if backend is None:
        print("Input not recognized. Please enter a number from 1 to 4, or 'all'.")
    else:
        with open(output_file, 'a') as f:
            f.write(f"\n\n-- Running tests with backend: {backend} --\n")
        subprocess.run(f"KERAS_BACKEND={backend} pytest {test_file} >> {output_file} 2>&1", shell=True)

# Final: Write the help message
with open(output_file, 'a') as f:
    f.write("\n\nPlease review the failed tests and provide suggestions to improve the functions for better results.")

print(f"Test results and file contents have been generated and appended to {output_file}.")
