import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Reshape, Concatenate


def load_mnist():
    """ Tensorflow Keras Default MNIST Data Loader

    Returns:
        (ndarray): MNIST digit images
        (ndarray): MNIST digit image classes

    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    return x_all, y_all


def get_tiled_image(images, labels, w, h, n):
    """

    Args:
        images (ndarray): source images
        labels (ndarray): detection labels
        w (int): width of digit image
        h (int): height of digit image
        n (int): size of detection map

    Returns:
        (ndarray): concatenated source images
        (ndarray): mapped detection labels

    """
    x = np.zeros((w * n, h * n))
    y = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            index = i + j * n
            x[i * w: (i + 1) * w, j * h: (j + 1) * h] = images[index]
            if labels is not None:
                y[i, j] = int(labels[index])
    return x, y


def load_mnist_detection(n=2, num_samples=10_000):
    """

    Args:
        n (int): size of detection map
        num_samples (int): number of samples to train or eval

    Returns:
        (ndarray): the list of image with shape (n x n), each element is randomly selected images (28 x 28)
        (ndarray): the list of one-hot encoded feature to target a specific digit
        (ndarray): the list of label with shape (n x n), each element is binaries (0 or 1)
        (list): the list of detection digit number

    """
    images, labels = load_mnist()
    num_detection_labels = n * n
    label_indices = defaultdict(lambda: [])
    for i, l in enumerate(labels):
        label_indices[l].append(i)
    pbar = tqdm(num_samples, desc='Initializing data for MNIST Detection Problem')
    num_mnist = len(images)
    num_classes = len(label_indices.keys())
    x1_all = []
    x2_all = []
    y_all = []
    y_labels = []
    for _ in range(num_samples):
        sample_indices = np.random.randint(0, num_mnist, num_detection_labels)
        sample_images = images[sample_indices]
        sample_labels = labels[sample_indices]
        target_label = np.random.choice(sample_labels, 1)[0]
        w = sample_images[0].shape[0]
        h = sample_images[0].shape[1]
        x1, y = get_tiled_image(sample_images, sample_labels, w, h, n)
        x2 = np.zeros(num_classes)
        x2[target_label] = 1
        x1 = np.expand_dims(x1, axis=-1)
        y = y == target_label
        y = y.astype(int)
        x1_all.append(x1)
        x2_all.append(x2)
        y_all.append(y)
        y_labels.append(target_label)
        pbar.update()

    x1_all = np.array(x1_all)
    x2_all = np.array(x2_all)
    y_all = np.array(y_all)

    # just for example ...
    plt.matshow(x1)
    plt.savefig('imgs/x1.png')

    return x1_all, x2_all, y_all, y_labels


def cnn_detector_4x4(input_1_shape, input_2_shape):
    """

    Args:
        input_1_shape (tuple): template image shape
        input_2_shape (tuple): target (detection) digit feature shape

    Returns:
        (tensorflow.keras.Model): CNN based 4x4 MNIST detection model

    """
    input_1 = tf.keras.layers.Input(shape=input_1_shape)
    input_2 = tf.keras.layers.Input(shape=input_2_shape)
    x1 = Conv2D(filters=32, kernel_size=4, padding='same', strides=1, activation='relu')(input_1)  # 112,112,1
    x1 = Conv2D(filters=32, kernel_size=4, padding='same', strides=2, activation='relu')(x1)  # 56,56,32
    x1 = Conv2D(filters=32, kernel_size=4, padding='same', strides=2, activation='relu')(x1)  # 28,28,32
    x1 = Conv2D(filters=32, kernel_size=4, padding='same', strides=2, activation='relu')(x1)  # 14,14,32

    # TODO: How to concatenate onehot label into the image feature map?
    x2 = Dense(14 * 14 * 32, activation='relu')(input_2)
    x2 = Reshape((14, 14, 32))(x2)
    x = Concatenate(axis=-1)([x1, x2])

    x = Conv2D(filters=32, kernel_size=4, padding='same', strides=2, activation='relu')(x)  # 7,7,32
    y = Conv2D(filters=1, kernel_size=4, padding='same', strides=2, activation='sigmoid')(x)  # 4,4,1
    y = Reshape((4, 4))(y)
    model = Model([input_1, input_2], y, name="MNIST-Detector")
    model.summary()
    return model


def main():
    """ MNIST Detection
        (baseline, maximum accuracy approx. 0.6 + @, current version is not stable)

        Input
            - Query [10]
            - Target Image NxN tiled digit images [(28 x N) x (28 x N)]

        Output
            - Positional Detection Probabilities [N x N]

        Model (N = 4 only)
            - Convolutional Neural Network Layers & DNN Layers
            - Convolutionalization (?)

        Dataset
            - MNIST digits 0 to 9 (10 classes)
            - N : 4

    """
    n = 4
    x1_train, x2_train, y_train, label_train = load_mnist_detection(n=n, num_samples=20_000)
    x1_test, x2_test, y_test, label_test = load_mnist_detection(n=n, num_samples=1_000)
    inp_1_shape = x1_train[0].shape
    inp_2_shape = x2_train[0].shape
    model = cnn_detector_4x4(inp_1_shape, inp_2_shape)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit([x1_train, x2_train], y_train, epochs=20)
    model.evaluate([x1_test, x2_test], y_test, verbose=1)
    num_eval_samples = 10

    for i, (x1, x2, y, label) in enumerate(zip(x1_test, x2_test, y_test, label_test)):
        x1 = np.array([x1])
        x2 = np.array([x2])
        p = model.predict([x1, x2])
        probabilities = p[0]
        img = x1[0]
        heatmap = np.zeros_like(img)
        for r in range(n):
            for c in range(n):
                heatmap[r * 28: (r + 1) * 28, c * 28: (c + 1) * 28] = probabilities[r, c]

        plt.matshow(img)
        plt.title('Detect: {}'.format(label))
        plt.savefig('imgs/test_sample_{}.png'.format(i + 1))

        plt.matshow(heatmap)
        plt.colorbar()
        plt.title('Detect: {} / prediction heatmap'.format(label))
        plt.savefig('imgs/test_sample_{}_detect_heatmap.png'.format(i + 1))

        if i >= num_eval_samples:
            break


if __name__ == '__main__':
    main()
