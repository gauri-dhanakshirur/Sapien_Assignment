# Image Classifier
A lightweight Multi-Layer Perceptron (MLP) built strictly with NumPy to classify images from the CIFAR-10 dataset without using deep learning frameworks like PyTorch or TensorFlow.

Dataset & Target Classes
Filters the original CIFAR-10 dataset down to 3 classes:
1. airplane (Class 0)
2. automobile (Class 1)
3. bird (Class 2)

Installation & Dependencies
Requires Python 3.8+ and the following packages:
pip install numpy matplotlib torchvision

Note: torchvision is used exclusively as a fast dataset mirror loader and is not involved in any model computation.

How to Run
Open Google Colab or a local Jupyter Notebook.
Run the main training script.
Model evaluation, loss curves, and confusion matrix will render automatically upon completion.
