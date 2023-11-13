import os
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.DeepNeuron import DeepNeuron
from src.FullyConnectedLayer import FullyConnectedLayer
from src.ActivationLayer import ActivationLayer
from src.Activations import RELU, RELU_PRIME,SIGMOID, SIGMOID_PRIME
from src.Losses import BINARY_CROSS_ENTROPY,BINARY_CROSS_ENTROPY_PRIME


# Function to download and extract the Optdigits dataset
def GetDigitData():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
    file_path = "optdigits.tra"

    # Download the dataset
    if not os.path.exists(file_path):
        print("Downloading Optdigits dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")


# Function to preprocess the data and save it into input and target files
def PreProcessSaveData():
    # Load the dataset
    data = np.loadtxt("optdigits.tra", delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the preprocessed data into input and target files
    np.save("../train/X_train.npy", X_train)
    np.save("../train/y_train.npy", y_train)
    np.save("../test/X_test.npy", X_test)
    np.save("../test/y_test.npy", y_test)


def GetTrainigData():
    return np.load("../train/X_train.npy"), np.load("../train/y_train.npy"), np.load("../test/X_test.npy"), np.load(
        "../test/y_test.npy")

def GetModelArch():
    model = DeepNeuron()
    model.ADD_LAYERS(FullyConnectedLayer(64, 64))
    model.ADD_LAYERS(ActivationLayer(RELU, RELU_PRIME))
    model.ADD_LAYERS(FullyConnectedLayer(64, 128))
    model.ADD_LAYERS(ActivationLayer(RELU, RELU_PRIME))
    model.ADD_LAYERS(FullyConnectedLayer(128, 32))
    model.ADD_LAYERS(ActivationLayer(RELU, RELU_PRIME))
    model.ADD_LAYERS(FullyConnectedLayer(32, 16))
    model.ADD_LAYERS(ActivationLayer(RELU, RELU_PRIME))
    model.ADD_LAYERS(FullyConnectedLayer(16, 10))
    model.ADD_LAYERS(ActivationLayer(SIGMOID, SIGMOID_PRIME))
    return model

def training(model,X_train,y_train):
    model.LOSSES(BINARY_CROSS_ENTROPY, BINARY_CROSS_ENTROPY_PRIME)
    model.FIT(X_train, y_train, epochs=10, learning_rate=0.01)

def main():
    GetDigitData()
    PreProcessSaveData()
    print('Data Extraction Completed!!!')
    X_train, y_train, X_test, y_test = GetTrainigData()
    #print(X_train[0],y_train.reshape(1,-1))
    print('The shape of training and test data =>', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = GetModelArch()
    print('===Starting Deep Neural Network Training===')
    training(model,X_train,y_train.reshape(1,-1))


if __name__ == '__main__':
    main()
