from algos.games import Evaluator, CompositeEvaluator, GameAlgo, GameConfig, GameState, Value

class Monotonic(Evaluator):
    def __init__(self, emptyWeight=50, mergeableWeight=50, montonicWeight=100, totalValueWeight=100):
        self.minValue = float('-inf')
        self.maxValue = float('inf')
        self.emptyWeight = emptyWeight
        self.mergeableWeight = mergeableWeight
        self.montonicWeight = montonicWeight
        self.totalValueWeight = totalValueWeight

    def __call__(self,state:GameState) -> Value:
        """
        TODO:
        figure out best way to impose smoothness
        try non-snake pattern, weighting a corner heavily
        """
        baseValue = 1000
        mergeable = 0
        empty = 0
        increasingLeft = 0
        increasingRight = 0
        totalValue = 0

        values = list(range(state.state.size))
        rows = [r for r in state.state.map]
        columns = [[state.state.map[i][j] for i in values]
                for j in values ]

        for i, row in enumerate(columns + rows):
            last = None
            for cell in row:
                totalValue += cell*cell
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                if last:
                    if cell < last:
                        increasingLeft += (last - cell)*(last - cell)*(last - cell)
                    else:
                        increasingRight += (cell - last)*(cell - last)*(cell - last)
                last = cell

        if empty == 0:
            return -100000

        return (
            baseValue
            + mergeable*self.mergeableWeight
            + empty*self.emptyWeight
            + totalValue*self.totalValueWeight
            - min(increasingLeft, increasingRight)*self.montonicWeight
        )
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue


class Snake(Evaluator):
    def __init__(self, emptyWeight=50, mergeableWeight=50, snakeWeight=50, totalValueWeight=50):
        self.minValue = float('-inf')
        self.maxValue = float('inf')
        self.emptyWeight = emptyWeight
        self.mergeableWeight = mergeableWeight
        self.totalValueWeight = totalValueWeight
        self.snakeWeight = snakeWeight

    def __call__(self,state:GameState) -> Value:
        """
        TODO:
        figure out best way to impose smoothness
        try non-snake pattern, weighting a corner heavily
        """
        baseValue = 1000
        mergeable = 0
        empty = 0
        snake = 0
        cellWeight = 4
        totalValue = 0

        values = list(range(state.state.size))
        rows = [r for r in state.state.map]
        cols = [[state.state.map[i][j] for i in values] for j in values ]

        reverseIt = False
        for row in rows:
            last = None
            rowIter = reversed(row) if reverseIt else row
            for cell in rowIter:
                totalValue += cell * cell
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                    last = cell
                snake += cell * cellWeight
                cellWeight *= 4
            reverseIt = not reverseIt

        for col in cols:
            last = None
            for cell in col:
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                    last = cell


        if empty == 0:
            return -100000

        return (
            baseValue
            + mergeable*self.mergeableWeight
            + empty*self.emptyWeight
            + totalValue*self.totalValueWeight
            + snake*self.snakeWeight
        )
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue



class Corner(Evaluator):
    def __init__(self, emptyWeight=50, mergeableWeight=50, cornerWeight=50, totalValueWeight=50):
        self.minValue = float('-inf')
        self.maxValue = float('inf')
        self.emptyWeight = emptyWeight
        self.mergeableWeight = mergeableWeight
        self.cornerWeight = cornerWeight
        self.totalValueWeight = totalValueWeight

    def __call__(self,state:GameState) -> Value:
        """
        TODO:
        figure out best way to impose smoothness
        try non-snake pattern, weighting a corner heavily
        """
        baseValue = 1000
        mergeable = 0
        empty = 0
        corner = 0
        cellWeight = 4
        totalValue = 0

        values = list(range(state.state.size))
        rows = [r for r in state.state.map]
        cols = [[state.state.map[i][j] for i in values] for j in values ]

        for row in rows:
            last = None
            for cell in row:
                totalValue += cell * cell
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                    last = cell
                corner += cell * cellWeight
                cellWeight *= 4

        cellWeight = 4
        for col in cols:
            last = None
            for cell in col:
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                    last = cell
                corner += cell * cellWeight
                cellWeight *= 4

        if empty == 0:
            return -100000

        return (
            baseValue
            + mergeable*self.mergeableWeight
            + empty*self.emptyWeight
            + corner*self.cornerWeight
        )
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue