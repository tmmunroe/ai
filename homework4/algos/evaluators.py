from algos.games import Evaluator, CompositeEvaluator, GameAlgo, GameConfig, GameState, Value

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
