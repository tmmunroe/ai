from algos.games import Evaluator, CompositeEvaluator, GameAlgo, GameConfig, GameState, Value

class Monotonic(Evaluator):
    """primarily informed by stackoverflow article referenced in homework, particularly the approach to
        applying monotonicity as a penalty that scales with the size of the tiles involved
       other than weighting, the main difference is that I add the total value measure to the score
         rather than subtracting it
       I return -inf if the board is unplayable
       'emptyWeight': 695, 'mergeableWeight': 812, 'montonicWeight': 727, 'totalValueWeight': 617
       """
    def __init__(self, emptyWeight=695, mergeableWeight=812, montonicWeight=727, totalValueWeight=617):
        self.minValue = float('-inf')
        self.maxValue = float('inf')
        self.emptyWeight = emptyWeight
        self.mergeableWeight = mergeableWeight
        self.montonicWeight = montonicWeight
        self.totalValueWeight = totalValueWeight

    def __str__(self):
        return f"Monotonic(emptyWeight={self.emptyWeight}, mergeableWeight={self.mergeableWeight}, montonicWeight={self.montonicWeight}, totalValueWeight={self.totalValueWeight})"

    def __call__(self,state:GameState) -> Value:
        baseValue = 10000
        mergeable = 0
        empty = 0
        nonMonotonicity = 0
        totalValue = 0
        value = 0

        values = list(range(state.state.size))
        rows = [r for r in state.state.map]
        columns = [[state.state.map[i][j] for i in values]
                for j in values ]

        for row in columns + rows:
            """
            print()
            print(f"{row}")
            """
            last = None
            increasingLeft = 0
            increasingRight = 0
            for cell in row:
                totalValue += pow(cell,2)
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                if last is not None:
                    if cell < last:
                        increasingLeft += pow(last,2) - pow(cell,2)
                    else:
                        increasingRight += pow(cell,2) - pow(last,2)
                    """print(f"cell:{cell} last: {last} --> left: {increasingLeft} right: {increasingRight}")"""
                last = cell
            nonMonotonicity += min(increasingLeft, increasingRight)
            """print(f"nonmonotonicity: {min(increasingLeft, increasingRight)} = min({increasingLeft}, {increasingRight})")"""

        if empty == 0 and mergeable == 0:
            return float('-inf')

        #print(f"{baseValue} + {mergeable*self.mergeableWeight} + {empty*self.emptyWeight} + {totalValue*self.totalValueWeight} - {nonMonotonicity*self.montonicWeight}")
        return (
            baseValue
            + mergeable*self.mergeableWeight
            + empty*self.emptyWeight
            + totalValue*self.totalValueWeight
            - nonMonotonicity*self.montonicWeight
        )
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue


class Snake(Evaluator):
    def __init__(self, emptyWeight=50, mergeableWeight=50, snakeWeight=50, totalValueWeight=50, unsmoothPenaltyWeight=50):
        self.minValue = float('-inf')
        self.maxValue = float('inf')
        self.emptyWeight = emptyWeight
        self.mergeableWeight = mergeableWeight
        self.totalValueWeight = totalValueWeight
        self.snakeWeight = snakeWeight
        self.unsmoothPenaltyWeight = unsmoothPenaltyWeight

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
        unsmoothPenalty = 0

        values = list(range(state.state.size))
        rows = [r for r in state.state.map]
        cols = [[state.state.map[i][j] for i in values] for j in values ]

        reverseIt = False
        for row in rows:
            last = None
            rowIter = reversed(row) if reverseIt else row
            for cell in rowIter:
                totalValue += pow(cell,2)
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                if last is not None and cell < last:
                    unsmoothPenalty += pow(last,4) - pow(cell,4)
                last = cell
                snake += cell * cellWeight
                cellWeight *= 4
            reverseIt = not reverseIt

        for col in reversed(cols):
            last = None
            for cell in col:
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                if last is not None and cell < last:
                    unsmoothPenalty += pow(last,4) - pow(cell,4)
                last = cell


        if empty == 0 and mergeable == 0:
            return float('-inf')

        #print(f"{baseValue} + {mergeable*self.mergeableWeight} + {empty*self.emptyWeight} + {totalValue*self.totalValueWeight} + {snake*self.snakeWeight} - {unsmoothPenalty*self.unsmoothPenaltyWeight}")
        return (
            baseValue
            + mergeable*self.mergeableWeight
            + empty*self.emptyWeight
            + totalValue*self.totalValueWeight
            + snake*self.snakeWeight
            - unsmoothPenalty*self.unsmoothPenaltyWeight
        )
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue


class Corner(Evaluator):
    def __init__(self, emptyWeight=800, mergeableWeight=800, cornerWeight=800, totalValueWeight=200, unsmoothPenaltyWeight=800):
        self.minValue = float('-inf')
        self.maxValue = float('inf')
        self.emptyWeight = emptyWeight
        self.mergeableWeight = mergeableWeight
        self.cornerWeight = cornerWeight
        self.totalValueWeight = totalValueWeight
        self.unsmoothPenaltyWeight = unsmoothPenaltyWeight

    def __str__(self):
        return f"Corner(emptyWeight={self.emptyWeight}, mergeableWeight={self.mergeableWeight}, cornerWeight={self.cornerWeight}, totalValueWeight={self.totalValueWeight}, unsmoothPenaltyWeight={self.unsmoothPenaltyWeight})"

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
        cellWeight = 2
        totalValue = 0
        unsmoothPenalty = 0

        values = list(range(state.state.size))
        rows = [r for r in state.state.map]
        cols = [[state.state.map[i][j] for i in values] for j in values ]

        for row in rows:
            last = None
            for cell in reversed(row):
                totalValue += pow(cell, 2)
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                if last is not None and cell < last:
                    unsmoothPenalty +=  (last-cell) * cellWeight
                last = cell
                corner += cell * cellWeight
                cellWeight *= 2

        for col in cols:
            last = None
            for cell in col:
                totalValue += pow(cell, 2)
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                if last is not None and cell < last:
                    unsmoothPenalty +=  (last-cell) * cellWeight
                last = cell

        if empty == 0 and mergeable == 0:
            return float('-inf')

        #print(f"{baseValue} + {mergeable*self.mergeableWeight} + {empty*self.emptyWeight} + {corner*self.cornerWeight} - {unsmoothPenalty*self.unsmoothPenaltyWeight}")
        return (
            baseValue
            + mergeable*self.mergeableWeight
            + empty*self.emptyWeight
            + corner*self.cornerWeight
            + totalValue*self.totalValueWeight
            - unsmoothPenalty*self.unsmoothPenaltyWeight
        )
    
    @property
    def MinValue(self):
        return self.minValue

    @property
    def MaxValue(self):
        return self.maxValue