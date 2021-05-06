import Grid
from typing import NewType, Sequence

MaxAction = NewType("MaxAction", int)
MinAction = NewType("MinAction", int)

class EfficientGrid:
    def __init__(self, grid: Grid.Grid):
        pass

    def maxActions(self) -> Sequence[MaxAction]:
        pass

    def applyMax(self, action: MaxAction) -> EfficientGrid:
        pass

    def minActions(self) -> Sequence[MinAction]:
        pass

    def applyMin(self, action: MinAction) -> EfficientGrid:
        pass
    