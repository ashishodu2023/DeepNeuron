from src.Layer import Layer

# inherit from base class Layer
class FlattenLayer(Layer):
    # returns the flattened input
    def FORWARD(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def BACKWORD(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)