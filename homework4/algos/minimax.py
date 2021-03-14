from algos.games import GameAlgo, GameState, NullAction, Value, Action
from typing import Tuple
import Grid


class Minimax(GameAlgo):
    def maximize(self, state:GameState, depth:int) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MinValue

        actionStates = state.maxMoves()
        self.stats.addBranchFactor(len(actionStates))
        for action, state in actionStates:
            _, minValue = self.minimize(state, depth-1)
            if minValue > value:
                best, value = action, minValue
        
        return best, value

    def minimize(self, state:GameState, depth:int) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MaxValue

        actionStates = state.minMoves()
        self.stats.addBranchFactor(len(actionStates))
        for action, state in actionStates:
            _, maxValue = self.maximize(state, depth-1)
            if maxValue < value:
                best, value = action, maxValue
        
        return best, value

    def search(self) -> Action:
        depth = 1
        while True:
            best, value = self.maximize(self.initialState, depth)
            self.checkAndSetAction(best, value)
            if self.searchTerminated():
                break
            depth += 1
        self.stats.addSearchDepth(depth)
        return self.bestAction()
