import numpy as np
import numpy.typing as npt
from typing import Callable

Vector = npt.ArrayLike
Matrix = npt.ArrayLike
Measure = Callable[[Vector, Vector], float]

def manhattanDistance(a:Vector, b:Vector) -> float:
    diffs = np.subtract(a,b)
    return sum(np.absolute(diffs))


def euclideanDistance(a:Vector, b:Vector) -> float:
    return np.dot(a, b)