# Project Name

Short description or overview of the project.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [Models](#models)
5. [Optimizers](#optimizers)
6. [Learning Rate Schedulers](#learning-rate-schedulers)
7. [Contributing](#contributing)
8. [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_project.git

2. Install dependencies:

    ```bash
    pip install tensorflow keras scikit-learn tqdm pandas itertools 

## Usage

1. Download the dataset and place it in the appropriate directory.
2. Update the dataset path and other configurations in `config.yaml`.
3. Execute `final.py` to start the training process.
4. The trained model will be saved in the `Models` directory.

## Configuration

The `config.yaml` file contains the following configurations:

- **Dataset Path**: Specify the path to the dataset directory.

- **Categories**: List of categories/classes present in the dataset. If the list is empty, all categories will be considered for model training.

- **Epochs**: Number of training epochs.

- **Batch Size**: Batch size used during training.

- **Model Name**: Name of the pre-trained model to use for transfer learning.

- **Optimizer**: Type of optimizer to use during model training (e.g., Adam, SGD).

- **Learning Rate**: Learning rate for the optimizer.

- **Learning Rate Scheduler**: Type of learning rate scheduler to adjust learning rates during training (e.g., Exponential decay, Inverse time decay).

- **Model File Name**: File path for saving the trained model in HDF5 format (e.g., "Models/model1.h5").

- **TFLite File Name**: File path for saving the trained model in TensorFlow Lite format (e.g., "Models/tflite1.tflite").

Update these configurations as needed before running the code.

## Models

The project supports the following pre-trained models for transfer learning:

- VGG16
- VGG19
- ResNet50
- ResNet101
- ResNet152
- InceptionV3
- Xception
- MobileNet
- MobileNetV2
- DenseNet121
- DenseNet169
- DenseNet201
- NASNetMobile
- NASNetLarge


## Optimizers

The project supports the following optimizers:

- Adam
- Adadelta
- Adagrad
- RMSprop
- SGD

## Learning Rate Schedulers

The project supports the following learning rate schedulers:

- Exponential decay
- Inverse time decay
- Polynomial decay
- Piecewise constant decay

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.
