import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn.utils import shuffle
import h5py


class NeuralNet:
    """Private functions"""

    def __initialize_parameters(self, layer_dims: list):
        np.random.seed(1)
        parameters = {}
        Layers = len(layer_dims)

        for l in range(1, Layers):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        return parameters

    def __compute_cost(self, AL: np.array, Y: np.array):
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
        return np.squeeze(cost)

    def __L_model_forward(self, X, parameters):
        def linear_activation_forward(A_prev: np.array, W: np.array, b: np.array, activation='sigmoid'):
            def linear_forward(A, W, b):
                Z = np.dot(W, A) + b
                cache = (A, W, b)
                return Z, cache

            def sigmoid(Z):
                A = 1 / (1 + np.exp(-Z))
                cache = (Z)
                return A, cache

            def relu(Z):
                A = np.maximum(Z, 0, Z)
                cache = (Z)
                return A, cache

            if activation == 'sigmoid':
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = sigmoid(Z)
            elif activation == 'relu':
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = relu(Z)

            cache = (linear_cache, activation_cache)

            return A, cache

        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                                 activation='relu')
            caches.append(cache)

        AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],
                                              activation='sigmoid')
        caches.append(cache)

        return AL, caches

    def __L_model_backward(self, AL: np.array, Y: np.array, caches: tuple):
        def linear_activation_backward(dA: np.array, cache: np.array, activation: np.array):
            def linear_backward(dZ: np.array, cache: tuple):
                A_prev, W, b = cache
                m = A_prev.shape[1]

                dW = (1 / m) * np.dot(dZ, A_prev.T)
                db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
                dA_prev = np.dot(W.T, dZ)

                return dA_prev, dW, db

            def sigmoid_backward(dA: np.array, cache: np.array):
                Z = cache
                s = 1 / (1 + np.exp(-Z))
                dZ = dA * s * (1 - s)
                return dZ

            def relu_backward(dA: np.array, cache: np.array):
                Z = cache
                dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
                # When z <= 0, you should set dz to 0 as well.
                dZ[Z <= 0] = 0
                return dZ

            linear_cache, activation_cache = cache
            if activation == 'sigmoid':
                dZ = sigmoid_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)
            elif activation == 'relu':
                dZ = relu_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)

            return dA_prev, dW, db

        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
        grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)] = dA_prev_temp, dW_temp, db_temp

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dQ_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
            grads['dA' + str(l)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = dA_prev_temp, dQ_temp, db_temp

        return grads

    def __update_parameters(self, params, grads, learning_rate):
        parameters = params.copy()
        L = len(parameters) // 2

        for l in range(L):
            parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
            parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

        return parameters

    def __init__(self, layer_dims: list, learning_rate: float, num_iterations: int, print_cost: bool):
        self.__costs = []
        self.__layers_dims = layer_dims
        self.__learning_rate = learning_rate
        self.__num_iterations = num_iterations
        self.__print_cost = print_cost

    def fit(self, X: np.array, Y: np.array):
        np.random.seed(1)
        self.__parameters = self.__initialize_parameters(self.__layers_dims)
        # print('Parameters: ', self.__parameters)
        for i in range(0, self.__num_iterations):
            AL, caches = self.__L_model_forward(X, self.__parameters)
            # print('AL: ', AL)
            # print('Cache: ', caches)
            cost = self.__compute_cost(AL, Y)
            # print('Cost: ', cost)
            grads = self.__L_model_backward(AL, Y, caches)
            # print('Grads: ', grads)
            self.__parameters = self.__update_parameters(self.__parameters, grads, self.__learning_rate)
            # print('New Parameters: ', self.__parameters)

            if self.__print_cost and i % 100 == 0 or i == self.__num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == self.__num_iterations:
                self.__costs.append(cost)

    def predict(self, X: np.array, y: np.array):
        m = X.shape[1]
        n = len(self.__parameters) // 2
        p = np.zeros((1, m))

        probas, cache = self.__L_model_forward(X, self.__parameters)

        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print('Accuracy: ', np.sum((p == y)/m))

    def print_mislabeled_images(self, classes, X, y, p):
        a = p + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
        num_images = len(mislabeled_indices[0])
        for i in range(num_images):
            index = mislabeled_indices[1][i]

            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[
                y[0, index]].decode("utf-8"))

def load_dataset(folder_path, is_train):
    path = folder_path
    data = []
    labels = []
    for filename in glob.glob(path):
        if is_train is True:
            if filename.find('cat') != -1:
                img = cv2.imread(filename)
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                data.append(np.array(img))
                labels.append(1)
        else:
            img = cv2.imread(filename)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            data.append(np.array(img))
            labels.append(np.inf)

    data = np.array(data)
    labels = np.array(labels).reshape((len(labels), 1))
    data, labels = shuffle(data, labels)
    return data, labels


def load_data_from_h5py():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == '__main__':

    option = int(input('Press 0 to load data from images, 1 for h5py: '))
    if option == 0:
        start = time.time()
        X_train, y_train = load_dataset('train/*.jpg', True)
        X_test, y_test = load_dataset('test1/*.jpg', False)

        '''Flattening and standardizing'''
        X_train = X_train.reshape(X_train.shape[0], -1).T / 255
        X_test = X_test.reshape(X_test.shape[0], -1).T / 255
        y_train = y_train.T
        y_test = y_test.T

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        print('Time taken to load images: ', time.time() - start)
    else:
        start = time.time()
        X_train, y_train, X_test, y_test, classes = load_data_from_h5py()

        X_train = X_train.reshape(X_train.shape[0], -1).T / 255
        X_test = X_test.reshape(X_test.shape[0], -1).T / 255

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        print('Time taken to load h5py: ', time.time() - start)

    layers_dims = [12288, 20, 7, 5, 1]
    model = NeuralNet(layers_dims, 0.0075, 2500, True)
    model.fit(X_train, y_train)
    print('Time taken: ', (time.time() - start))
    print('Train ', end='')
    model.predict(X_train, y_train)
    print('Test ', end='')
    model.predict(X_test, y_test)
    # Putting dog image
    img = cv2.imread('a.jpg')
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = np.array(img)
    # print(img.shape)
    img = img.reshape(-1, 1) / 255
    # print(img.shape)
    model.predict(img, [0])
