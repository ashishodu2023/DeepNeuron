from src.DeepNeuron import Network
from src.ActivationLayer import ActivationLayer
from src.FullyConnectedLayer import FCLayer
from src.Activations import relu, relu_derivative, softmax, softmax_derivative,tanh, tanh_derivative
from src.DataProcessing import DataProcessing
from src.Losses import  cross_entropy,cross_entropy_derivative

def GetModelArch():
    model = Network()
    model.add(FCLayer(1024, 1024))
    model.add(ActivationLayer(relu, relu_derivative))
    #model.add(FCLayer(64, 128))
    #model.add(ActivationLayer(relu, relu_derivative))
    #model.add(FCLayer(128, 32))
    #model.add(ActivationLayer(relu, relu_derivative))
    #model.add(FCLayer(32, 16))
    #model.add(ActivationLayer(relu, relu_derivative))
    model.add(FCLayer(1024, 10))
    model.add(ActivationLayer(softmax, softmax_derivative))
    return model


def main():
    pred =[]
    dp = DataProcessing()
    X_train, y_train, X_test, y_test = dp.DataProcessing()
    print('The shape of training and test data =>', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model = GetModelArch()
    model.use(cross_entropy, cross_entropy_derivative)
    print('============Started Training===========')
    model.fit(X_train, y_train.reshape(-1,1), epochs=10, learning_rate=0.01)

    # test on 3 samples
    """
    predictions = model.predict(X_test)
    for values in predictions[0]:
        pred.append(values)
    print("\n")
    print("predicted values : ")
    print(pred, end="\n")
    print("true values : ")
    print(y_test)
    """

if __name__ == '__main__':
    main()
