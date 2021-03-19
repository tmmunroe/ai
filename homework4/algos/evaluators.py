from algos.games import Evaluator, CompositeEvaluator, GameAlgo, GameConfig, GameState, Value

class TwentyFortyEightHeuristic(Evaluator):
    def __init__(self, emptyWeight=50, mergeableWeight=50, montonicWeight=50, totalValueWeight=50):
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
                        increasingLeft += (last - cell)*(last - cell)
                    else:
                        increasingRight += (cell - last)*(cell - last)
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