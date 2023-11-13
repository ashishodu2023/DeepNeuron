from tqdm import tqdm_notebook
class DeepNeuron:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def ADD_LAYERS(self, layer):
        self.layers.append(layer)

    # set loss to use
    def LOSSES(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def PREDICT(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in tqdm_notebook(range(samples)):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.FORWARD(output)
            result.append(output)

        return result

    # train the network
    def FIT(self, x_train, y_train, epochs, learning_rate):
        # training loop
        for i in tqdm_notebook(range(epochs)):
            loss = 0
            for j in range(len(x_train)):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.FORWARD(output)

                # compute loss (for display purpose only)
                loss += self.loss(y_train[j], output)
                print(loss)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.BACKWORD(error, learning_rate)

            # calculate average error on all samples
            loss /= samples
            print('EPOCH %d/%d   LOSS=%f' % (i+1, epochs, loss))