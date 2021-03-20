from algos.games import Evaluator, CompositeEvaluator, GameAlgo, GameConfig, GameState, Value

class Monotonic(Evaluator):
    def __init__(self, emptyWeight=1000, mergeableWeight=500, montonicWeight=700, totalValueWeight=100):
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

        if empty == 0:
            return -100000

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
                    unsmoothPenalty += pow(last,3) - pow(cell,3)
                last = cell
                snake += cell * cellWeight
                cellWeight *= 4
            reverseIt = not reverseIt

        for col in cols:
            last = None
            cellWeight = 4
            for cell in col:
                if cell == 0:
                    empty += 1
                elif cell == last:
                    mergeable += 1
                if last is not None and cell < last:
                    unsmoothPenalty += pow(last,3) - pow(cell,3)
                last = cell
                snake += cell * cellWeight
                cellWeight *= 4


        if empty == 0:
            return -100000

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

        #print(f"{baseValue} + {mergeable*self.mergeableWeight} + {empty*self.emptyWeight} + {corner*self.cornerWeight}")

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