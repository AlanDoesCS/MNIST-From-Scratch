import os
import numpy as np
import struct
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(111)

# Matplotlib LaTeX configuration
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x: np.ndarray):
    shifted_x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def log_loss(output, target):
    m = target.shape[1]
    return -np.sum(target * np.log(output + 1e-9)) / m

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols) / 255.0  # Normalize to [0, 1]
    return images.reshape(num_images, -1).T

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    one_hot_labels = np.zeros((10, num_labels))
    for i, label in enumerate(labels):
        one_hot_labels[label, i] = 1
    return one_hot_labels

class NeuralNetwork:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.weights, self.biases = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        biases = []
        for l in range(1, len(self.layers)):
            W = np.random.randn(self.layers[l], self.layers[l - 1]) * np.sqrt(2 / self.layers[l - 1])
            b = np.zeros((self.layers[l], 1))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def forward_pass(self, X):
        activations_output = [X]
        linear_outputs = []
        for W, b, activation in zip(self.weights, self.biases, self.activations):
            Z = np.dot(W, activations_output[-1]) + b
            linear_outputs.append(Z)
            A = activation(Z)
            activations_output.append(A)
        return activations_output, linear_outputs

    def compute_gradients(self, X, Y):
        activations_output, linear_outputs = self.forward_pass(X)

        dWs = [None] * len(self.weights)
        dbs = [None] * len(self.biases)

        A_final = activations_output[-1]
        dZ = A_final - Y

        for l in reversed(range(len(self.weights))):
            A_prev = activations_output[l]
            dWs[l] = (1 / X.shape[1]) * np.dot(dZ, A_prev.T)  # Gradient of weights
            dbs[l] = (1 / X.shape[1]) * np.sum(dZ, axis=1, keepdims=True)  # Gradient of biases

            if l > 0:
                g_prime = relu_derivative(linear_outputs[l - 1])
                dZ = np.dot(self.weights[l].T, dZ) * g_prime

        return dWs, dbs

    def update_parameters(self, dWs, dbs, learning_rate):
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * dWs[l]
            self.biases[l] -= learning_rate * dbs[l]

    def train(self, X, Y, X_test, Y_test, learning_rate, epochs):
        test_accuracies = []

        for epoch in range(epochs):
            dWs, dbs = self.compute_gradients(X, Y)
            self.update_parameters(dWs, dbs, learning_rate)

            activations_output, _ = self.forward_pass(X_test)
            predictions = np.argmax(activations_output[-1], axis=0)
            true_labels = np.argmax(Y_test, axis=0)
            accuracy = np.mean(predictions == true_labels)
            test_accuracies.append(accuracy)

            if (epoch+1) % 10 == 0:
                loss = log_loss(activations_output[-1], Y_test)  # Use test labels
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

        return test_accuracies

if __name__ == "__main__":
    # Load MNIST dataset
    train_images_path = "../dataset/train-images.idx3-ubyte"
    train_labels_path = "../dataset/train-labels.idx1-ubyte"
    test_images_path = "../dataset/t10k-images.idx3-ubyte"
    test_labels_path = "../dataset/t10k-labels.idx1-ubyte"

    X_train = load_mnist_images(train_images_path)
    Y_train = load_mnist_labels(train_labels_path)
    X_test = load_mnist_images(test_images_path)
    Y_test = load_mnist_labels(test_labels_path)

    layers = [784, 128, 64, 10]  # Input layer is 28x28 px
    activations = [relu, relu, softmax]

    learning_rates = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512]
    epochs = 50

    accuracy_data = {}

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        nn = NeuralNetwork(layers, activations)
        test_accuracies = nn.train(X_train, Y_train, X_test, Y_test, learning_rate=lr, epochs=epochs)
        accuracy_data[lr] = test_accuracies

    df = pd.DataFrame(accuracy_data)
    df.index.name = "Epoch"
    df.to_csv("mnist_accuracy_per_epoch.csv")

    plt.figure(figsize=(12, 8))

    for lr, accuracies in accuracy_data.items():
        plt.plot(range(epochs), accuracies, label=f"LR={lr}")

    plt.title("Test Accuracy vs Epoch for Different Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)  # Fix the vertical range to [0, 1]
    plt.legend()
    plt.grid(True)
    plt.savefig("mnist_test_accuracy_vs_epoch.png")
    plt.show()
