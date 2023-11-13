import numpy as np

from src.DeepNeuron import DeepNeuron
from src.FullyConnectedLayer import FullyConnectedLayer
from src.ActivationLayer import ActivationLayer
from src.Activations import RELU, RELU_PRIME
from src.Losses import MSE, MSE_PRIME


def main():
    # training data
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    model = DeepNeuron()
    model.ADD_LAYERS(FullyConnectedLayer(2, 4))
    model.ADD_LAYERS(ActivationLayer(RELU, RELU_PRIME))
    model.ADD_LAYERS(FullyConnectedLayer(4, 1))
    model.ADD_LAYERS(ActivationLayer(RELU, RELU_PRIME))

    # train
    model.LOSSES(MSE, MSE_PRIME)
    model.FIT(x_train, y_train, epochs=1000, learning_rate=0.01)

    # test
    out = model.PREDICT(x_train)
    print(out)

if __name__=='__main__':
    main()