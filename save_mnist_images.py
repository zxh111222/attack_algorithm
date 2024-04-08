import os
import numpy as np
from PIL import Image


def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    magic_number, num_images, rows, cols = np.frombuffer(data[:16], dtype=np.dtype('>i4'), count=4)
    images = data[16:].reshape(num_images, rows, cols)
    return images


def save_images(images, labels, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (image, label) in enumerate(zip(images, labels)):
        image_path = os.path.join(save_dir, f"{label}_{i}.png")
        image = Image.fromarray(image)
        image.save(image_path)


# 读取训练图像和标签
train_images = read_mnist_images("./mnist/MNIST/raw/train-images-idx3-ubyte")
train_labels = np.fromfile("./mnist/MNIST/raw/train-labels-idx1-ubyte", dtype=np.uint8)[8:]

# 保存图像到目录
save_images(train_images, train_labels, "mnist_images")
