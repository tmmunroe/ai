import numpy as np
import numpy.typing as npt
from typing import Callable
import math

Vector = np.ndarray
Matrix = np.ndarray
Measure = Callable[[Vector, Vector], float]
Activation = Callable[[float], float]


def manhattanDistance(a:Vector, b:Vector) -> float:
    diffs = np.subtract(a,b)
    return sum(np.absolute(diffs))


def euclideanDistance(a:Vector, b:Vector) -> float:
    return np.dot(a, b)


def stepFunction(a: float) -> float:
    if a <= 0:
        return -1
    return 1

def sigmoid(a: float) -> float:
    return 1 / ( 1 + math.exp(-a))

def relu(a: float) -> float:
    return max(0, a)

def tanh(a: float) -> float:
    return math.tanh(a)
