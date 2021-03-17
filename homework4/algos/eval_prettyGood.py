from algos.games import Evaluator, CompositeEvaluator, GameAlgo, GameConfig, GameState, Value

class TwentyFortyEightHeuristic(Evaluator):
    def __init__(self, emptyWeight=0, mergeableWeight=0, montonicWeight=20, weightedCellsWeight=25):
        self.minValue = 0.0
        self.maxValue = float('inf')
        self.emptyWeight = emptyWeight
        self.mergeableWeight = mergeableWeight
        self.montonicWeight = montonicWeight
        self.weightedCellsWeight = weightedCellsWeight

    def __call__(self,state:GameState) -> Value:
        "empty and mergeable cells"
        baseValue = 1
        mergeable = 0
        empty = 0
        increasing = 0
        weightedCells = 0
        cellWeight = 1
        values = list(range(state.state.size))
        rows = [r for r in state.state.map]
        columns = [
            [state.state.map[i][j] for i in values]
                for j in reversed(values)
        ]
        reverseDirection = False
        for i, row in enumerate(rows + columns):
            last = None
            increased = True
            rowIter = row if not reverseDirection else reversed(row)
            for cell in rowIter:
                if cell == last:
                    mergeable += 1
                    if last and cell < last:
                        increased = False
                elif cell == 0:
                    empty += 1
                if i <= 3:
                    weightedCells += cellWeight * cell
                    cellWeight *= 4
                last = cell
            if increased:
                increasing += 1
            if i <= 3:
                #snake along rows, but always have highest weight in last column
                reverseDirection = not reverseDirection

        return (baseValue
            + mergeable*self.mergeableWeight
            + empty*self.emptyWeight
            + increasing*self.montonicWeight
            + weightedCells*self.weightedCellsWeight)
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue


class EmptySpaces(Evaluator):
    def __init__(self):
        self.minValue = 0
        self.maxValue = 1

    def __call__(self,state:GameState) -> Value:
        return len(state.state.getAvailableCells()) / (state.state.size * state.state.size)
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue


class MaxTile(Evaluator):
    def __init__(self, minTile=0, maxTile=4096):
        self.minValue = 0
        self.maxValue = 1
        self.minTile = minTile
        self.maxTile = maxTile

    def __call__(self,state:GameState) -> Value:
        return state.state.getMaxTile() / self.maxTile
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue


class MergeableTiles(Evaluator):
    def __init__(self):
        self.minValue = 0
        self.maxValue = 1
        self.minMergeable = 0
        self.maxMergeable = 24

    def __call__(self,state:GameState) -> Value:
        total = 0
        for i in range(state.state.size):
            for j in range(state.state.size):
                val = state.state.getCellValue((i,j))
                if state.state.crossBound((i,j+1)) and val == state.state.getCellValue((i,j+1)):
                    total += 1
                if state.state.crossBound((i+1,j)) and val == state.state.getCellValue((i,j+1)):
                    total += 1
        return total / self.maxMergeable
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue


class Monotonic(Evaluator):
    def __init__(self):
        self.minValue = 0
        self.maxValue = 1
        self.maxMonotonic = 24

    def __call__(self,state:GameState) -> Value:
        total = 0
        for i in range(state.state.size):
            for j in range(state.state.size):
                val = state.state.getCellValue((i,j))
                if val:
                    right = state.state.getCellValue((i,j+1))
                    if right and val <= right:
                        total += 1
                    down = state.state.getCellValue((i,j+1))
                    if down and val <= down:
                        total += 1

        return total / self.maxMonotonic
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue
