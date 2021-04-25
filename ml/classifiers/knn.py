import numpy as np
from typing import Any, Callable, Dict, List, Sequence, Tuple
from common import Vector, Matrix, Measure
from statistics import multimode

class KNN:
    def __init__(self, k: int, measureFunc:Measure):
        self.k: int = k
        self.measure: Measure = measureFunc
        self.data: Sequence = []
        self.labels: Sequence = []

    def kNearestNeighbors(self, instance:Any) -> Sequence:
        knn:List = [ (d,l,self.measure(instance, d)) for d,l in zip(self.data, self.labels) ]
        knn.sort(key= lambda tup: tup[2])
        return knn[:self.k]
    
    def __call__(self, instance:Sequence) -> Any:
        knn = self.kNearestNeighbors(instance)
        print(knn)
        labels = [ k[1] for k in knn ]
        return multimode(labels)[0]

    def loss(self, labels:Sequence, predictions:Sequence) -> float:
        totalLoss = 0
        for label, prediction in zip(labels, predictions):
            if label != prediction:
                #print(label, prediction)
                totalLoss += 1
        return totalLoss

    def train(self, samples:Sequence, labels:Sequence) -> Any:
        self.data = samples
        self.labels = labels
        predictions = [ self(sample) for sample in samples ]
        return self.loss(labels, predictions)

    def test(self, samples:Sequence, labels:Sequence) -> Any:
        predictions = [ self(sample) for sample in samples ]
        return self.loss(labels, predictions)