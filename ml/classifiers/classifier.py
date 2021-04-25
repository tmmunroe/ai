import pandas
from typing import Any, Protocol
from common import Vector


class Classifier(Protocol):
    def __call__(self, instance:Vector) -> Any:
        raise NotImplementedError("Classifiers must implement __call__")

    def loss(self, labels:Vector, predictions:Vector) -> float:
        raise NotImplementedError("Classifiers must implement loss")

    def train(self, samples:Vector, labels:Vector) -> Any:
        raise NotImplementedError("Classifiers must implement train")

    def test(self, samples:Vector, labels:Vector) -> Any:
        raise NotImplementedError("Classifiers must implement test")