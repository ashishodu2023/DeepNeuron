from src.DeepNeuron import Network
from src.ActivationLayer import ActivationLayer
from src.FullyConnectedLayer import FCLayer
from src.Activations import relu, relu_derivative, softmax, softmax_derivative
from src.Losses import cross_entropy, cross_entropy_derivative
from src.DataProcessing import DataProcessing
import numpy as np


def GetTrainigData():
    return np.load("../train/X_train.npy"), np.load("../train/y_train.npy"), np.load("../test/X_test.npy"), np.load(
        "../test/y_test.npy")


def GetModelArch():
    model = Network()
    model.add(FCLayer(64, 64))
    model.add(ActivationLayer(relu, relu_derivative))
    #model.add(FCLayer(64, 128))
    #model.add(ActivationLayer(relu, relu_derivative))
    #model.add(FCLayer(128, 32))
    #model.add(ActivationLayer(relu, relu_derivative))
    #model.add(FCLayer(32, 16))
    #model.add(ActivationLayer(relu, relu_derivative))
    model.add(FCLayer(64, 10))
    model.add(ActivationLayer(softmax, softmax_derivative))
    return model


def main():
    dp = DataProcessing()
    X_train, y_train, X_test, y_test = GetTrainigData()
    print('The shape of training and test data =>', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = GetModelArch()
    model.use(cross_entropy, cross_entropy_derivative)
    print('============Started Training===========')
    model.fit(X_train, y_train, epochs=100, learning_rate=0.1)

    # test on 3 samples
    """
    predictions = model.predict(X_test)
    print("\n")
    print("predicted values : ")
    print(predictions, end="\n")
    print("true values : ")
    print(y_test)
    """

if __name__ == '__main__':
    main()
