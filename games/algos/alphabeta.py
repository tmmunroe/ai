from algos.games import GameAlgo, GameState, NullAction, Value, Action
from typing import Tuple
import Grid


class AlphaBeta(GameAlgo):
    def maximize(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MinValue

        actionStates = state.maxMoves()
        self.sortMoves(actionStates)
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, state) in enumerate(actionStates):
            _, minValue = self.minimize(state, depth-1, alpha, beta)
            if minValue > value:
                best, value = action, minValue
                if value > alpha:
                    alpha = value
                    if alpha > beta:
                        self.stats.addPrunedBranches(len(actionStates) - i)
                        break
        
        return best, value

    def minimize(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MaxValue

        actionStates = state.minMoves()
        self.sortMoves(actionStates,reverse=True)
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, state) in enumerate(actionStates):
            _, maxValue = self.maximize(state, depth-1, alpha, beta)
            if maxValue < value:
                best, value = action, maxValue
                if value < beta:
                    beta = value
                    if alpha > beta:
                        self.stats.addPrunedBranches(len(actionStates) - i)
                        break
        
        return best, value

    def search(self) -> Action:
        depth = 1
        alpha = float('-inf')
        beta = float('inf')
        while True:
            best, value = self.maximize(self.initialState, depth, alpha, beta)
            self.checkAndSetAction(best, value)
            if self.searchTerminated():
                break
            depth += 1
        self.stats.addSearchDepth(depth)
        return self.bestAction()
