import numpy as np
from typing import Any, Callable, Dict, List, Tuple
from common import Vector, Matrix, Measure
from statistics import multimode

class KNN:
    def __init__(self, k: int, measureFunc:Measure):
        self.k: int = k
        self.measure: Measure = measureFunc
        self.data: List = []
        self.labels: List = []

    def kNearestNeighbors(self, instance:Any) -> List:
        knn:List = [ (data,label,self.measure(instance, data)) for data,label in zip(self.data, self.labels) ]
        knn.sort(key= lambda tup: tup[2])
        return knn[:self.k]
    
    def __call__(self, instance:Any) -> Any:
        knn = self.kNearestNeighbors(instance)
        labels = [ k[1] for k in knn ]
        return multimode(labels)[0]

    def loss(self, labels:List, predictions:List) -> float:
        totalLoss = 0
        for label, prediction in zip(labels, predictions):
            if label != prediction:
                #print(label, prediction)
                totalLoss += 1
        return totalLoss

    def train(self, samples:List, labels:List) -> Any:
        self.data = samples
        self.labels = labels
        predictions = [ self(sample) for sample in samples ]
        return self.loss(labels, predictions)

    def test(self, samples:List, labels:List) -> Any:
        predictions = [ self(sample) for sample in samples ]
        return self.loss(labels, predictions)