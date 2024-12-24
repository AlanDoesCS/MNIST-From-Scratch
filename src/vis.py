import numpy as np
import matplotlib.pyplot as plt
import struct

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def visualize_digits_in_order(images, labels):
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))

    for digit in range(10):
        idx = np.where(labels == digit)[0][0]
        axes[digit].imshow(images[idx], cmap='gray')
        axes[digit].set_title(f"Label: {digit}")
        axes[digit].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_file_path = "../dataset/train-images.idx3-ubyte"
    label_file_path = "../dataset/train-labels.idx1-ubyte"

    images = load_mnist_images(image_file_path)
    labels = load_mnist_labels(label_file_path)

    visualize_digits_in_order(images, labels)
