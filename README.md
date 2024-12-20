# COMPUTER-VISION

README for MNIST and Fashion MNIST Classifiers

Overview

This repository contains two deep learning models implemented using TensorFlow and Keras for classifying images from the MNIST and Fashion MNIST datasets. These models demonstrate fundamental concepts of neural networks, including preprocessing, model architecture, training, evaluation, and testing on new data.

Model 1: MNIST Data Classifier

Dataset

The MNIST dataset consists of grayscale images of handwritten digits (0-9) with dimensions 28x28.

Each image is labeled with the corresponding digit.

Steps in the Code

Dataset Loading:

tf.keras.datasets.mnist.load_data() is used to load the training and testing datasets.

Data Visualization:

Visualize example images with their labels using Matplotlib.

Normalization:

Pixel values are normalized to the range [0, 1] by dividing by 255.

Model Architecture:

A sequential neural network with the following layers:

Flatten layer to convert 28x28 images into a 1D array.

Dense layer with 100 neurons and ReLU activation.

Dense layer with 50 neurons and ReLU activation.

Dense layer with 10 neurons and Softmax activation for output.

Compilation:

Optimizer: Adam

Loss function: Sparse Categorical Crossentropy

Metrics: Accuracy

Training:

Model is trained for 10 epochs on the training dataset.

Evaluation:

Accuracy and loss are evaluated on the test dataset.

Classification report is generated using sklearn’s classification_report.

Incorrect Predictions:

Displays misclassified test images along with predicted and true labels.

Predict on New Data:

Loads a new image, preprocesses it to match the MNIST format, and predicts the digit.

Usage

Run the code in Google Colab or any Python environment with TensorFlow and Matplotlib installed. Ensure the new image for testing is provided at the specified path.

__________________________________________________________________________________________________________________________________________________________________________________________

Model 2: Fashion MNIST Data Classifier

Dataset

The Fashion MNIST dataset consists of grayscale images of clothing items (e.g., t-shirts, trousers, dresses) with dimensions 28x28.

Each image is labeled with its category (0-9).

Steps in the Code

Dataset Loading:

tf.keras.datasets.fashion_mnist.load_data() is used to load the training and testing datasets.

Data Visualization:

Visualize example images with their labels using Matplotlib.

Data Filtering:

Removes images with labels 8 and 6 from the dataset.

Normalization:

Pixel values are normalized to the range [0, 1] by dividing by 255.

Model Architecture:

A sequential neural network with the following layers:

Flatten layer to convert 28x28 images into a 1D array.

Dense layer with 128 neurons and ReLU activation.

Dense layer with 10 neurons and Softmax activation for output.

Compilation:

Optimizer: Adam

Loss function: Sparse Categorical Crossentropy

Metrics: Accuracy

Training:

Model is trained for 20 epochs on the filtered training dataset.

Evaluation:

Accuracy and loss are evaluated on the filtered test dataset.

Classification report is generated using sklearn’s classification_report.

Visualization of Class Examples:

Displays one image per class from the filtered training dataset.

Predict on New Data:

Loads a new image, preprocesses it to match the Fashion MNIST format, and predicts the category.

Usage

Run the code in Google Colab or any Python environment with TensorFlow and Matplotlib installed. Ensure the new image for testing is provided at the specified path.

__________________________________________________________________________________________________________________________________________________________________________________________
|requirements|

Python 3.x

TensorFlow

NumPy

Matplotlib

Scikit-learn

Install dependencies using:

pip install tensorflow numpy matplotlib scikit-learn
__________________________________________________________________________________________________________________________________________________________________________________________
|Running the Models|


Clone the repository or copy the code into your Python environment.

Ensure all dependencies are installed.

For both models, execute the code blocks sequentially.

Provide a path to the new image (for testing) in the respective section of the code.
______________________________________________________________________________________________________________________________________________________________________________________
|Results|


MNIST Model

Achieves high accuracy on test data.

Provides detailed classification report and visualizes misclassified samples.

Fashion MNIST Model

Handles filtered data by removing specific classes.

Visualizes class examples and provides detailed classification report.
______________________________________________________________________________________________________________________________________________________________________________________
|Customization|


Modify the number of layers or neurons to experiment with different architectures.

Adjust the number of epochs to fine-tune training performance.

Replace or augment datasets with additional images for extended testing.
______________________________________________________________________________________________________________________________________________________________________________________
|Contact|

For any questions or contributions, feel free to reach out via [jayanthkonanki82@gmail.com].

