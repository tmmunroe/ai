import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from common import Vector, Matrix, Measure, Activation

class Perceptron:
    def __init__(self, activation: Activation):
        self.activation = activation
        self.maxIterations = 100
        self.weights: Vector = np.array([])
        self.bias: float = 1
    
    def __call__(self, instance:Any) -> Any:
        vectorized = np.array(instance)
        weighted_sum = self.bias + np.dot(self.weights, vectorized)
        return self.activation(weighted_sum)

    def loss(self, labels:List, predictions:List) -> float:
        totalLoss = 0
        for label, prediction in zip(labels, predictions):
            if label != prediction:
                totalLoss += 1
        return totalLoss

    def train(self, data:List, labels:List) -> Any:
        sampleData = np.array(data)
        sampleLabels = np.array(labels)
        self.weights = np.ones(sampleData[0].shape)
        self.bias = 1

        iterations = 0
        while True:
            priorBias, priorWeights = self.bias, self.weights.copy()
            for x,y in zip(sampleData, sampleLabels):
                prediction = self(x)
                if prediction*y <= 0: #misclassified
                    self.bias += y
                    self.weights += x*y
            if self.bias == priorBias and all((w == prior_w for w, prior_w in zip(self.weights, priorWeights))):
                print('Converged')
                break
            if iterations > self.maxIterations:
                print('Exceeded max iterations.. Stopping')
                break

        predictions = [ self(sample) for sample in sampleData ]
        return self.loss(labels, predictions)

    def test(self, samples:List, labels:List) -> Any:
        predictions = [ self(sample) for sample in samples ]
        return self.loss(labels, predictions)