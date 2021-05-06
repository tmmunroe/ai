import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from common import Vector, Matrix, Measure, Activation

class MLPLayer(Matrix):
    def feature_weights(self, feature_index:int) -> Vector:
        return self[:,feature_index]

    def feature_error(self, errors:Vector, feature_index:int) -> float:
        feature_weights = self.feature_weights(feature_index)
        return np.dot(errors, feature_weights)

    def feed_forward(self, inputs: Vector, activation:Activation) -> Vector:
        outputs = np.dot(self, inputs, out=Vector)
        return np.array([activation(o) for o in outputs])

    def back_propagate(self, previous_outputs: Vector, errors: Vector) -> Vector:
        return previous_outputs*(1-previous_outputs)*errors

    def update_weights(self, learning_rate:float, inputs:Vector, errors:Vector):
        for row_error, row in zip(errors, self):
            row += learning_rate * row_error * inputs

class MultiLayerPerceptron:
    def __init__(self, activation: Activation, learning_rate:float, layers:int, nodes_per_layer:Vector):
        self.activation = activation
        self.layers = Vector([1])
        self.maxIterations = 100
        self.weights: Vector = np.array([])
        self.bias: float = 1
        self.learningRate: float = learning_rate

    def __call__(self, instance:Any) -> Any:
        inputs = np.array(instance)
        for layer in self.layers:
            outputs = layer.feed_forward(inputs)
            inputs = outputs
        return outputs

    def loss(self, labels:List, predictions:List) -> float:
        totalLoss = 0
        for label, prediction in zip(labels, predictions):
            if label != prediction:
                totalLoss += 1
        return totalLoss

    def train(self, data:List, labels:List) -> Any:
        sampleData = np.array(data)
        sampleLabels = np.array(labels)

        while True:
            for sample, label in zip(sampleData, sampleLabels):
                #feed forward - compute outputs for every layer
                layer_inputs = sample
                layer_outputs: List[Vector] = []
                last_layer_out: Vector = Vector(shape=label.shape)
                for layer in self.layers:
                    last_layer_out = layer.feed_forward(layer_inputs)
                    layer_outputs.append(last_layer_out)
                    layer_inputs = last_layer_out
                
                #back propagate - compute errors for every layer
                layer_errors: List[Vector] = []
                previous_layer_errors_weighted: Vector = label - last_layer_out
                for this_layer, this_layer_output in zip(np.flip(self.layers), reversed(layer_outputs)):
                    ones = np.ones(shape=this_layer_output.shape)
                    this_layer_errors = this_layer_output * (ones - this_layer_output) * previous_layer_errors_weighted
                    layer_errors.append(this_layer_errors)
                    previous_layer_errors_weighted = np.dot(this_layer.T, this_layer_errors)

                #update weights - update weights for every connection
                layer_inputs = [sample] + layer_outputs[:-1]
                for layer, layer_in, layer_err in zip(self.layers, layer_inputs, layer_errors):
                    for row, err in zip(layer, layer_err):
                        row = row + err * self.learningRate * layer_in
                pass
            
        predictions = [ self(sample) for sample in sampleData ]
        return self.loss(labels, predictions)

    def test(self, samples:List, labels:List) -> Any:
        predictions = [ self(sample) for sample in samples ]
        return self.loss(labels, predictions)