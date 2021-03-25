from algos.games import GameAlgo, GameState, NullAction, Value, Action
from typing import Any, Tuple, Dict, Sequence
import Grid


class ExpectiAlphaBeta(GameAlgo):
    def maximize(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MinValue

        actionStates = state.maxMoves()
        self.sortMoves(actionStates)
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, b) in enumerate(actionStates):
            _, minValue = self.expectedValue(b, depth-1, alpha, beta)
            if minValue >= value:
                best, value = action, minValue
                if value >= alpha:
                    alpha = value
                    if alpha >= beta:
                        self.stats.addPrunedBranches(len(actionStates) - i)
                        break
        
        return best, value

    def expectedValue(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        expectedValue = 0
        actionStates = state.chanceMoves()
        self.stats.addBranchFactor(len(actionStates))
        for probability, b in actionStates:
            _, value = self.minimize(b, depth-1, alpha, beta)
            expectedValue += probability * value

        return NullAction, expectedValue

    def minimize(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MaxValue

        """
        actionStates = state.minMoves()
        self.sortMoves(actionStates, reverse=True)
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, b) in enumerate(actionStates[:4]):
            _, maxValue = self.maximize(b, depth-1, alpha, beta)
            if maxValue <= value:
                best, value = action, maxValue
                if value <= beta:
                    beta = value
                    if alpha >= beta:
                        self.stats.addPrunedBranches(len(actionStates) - i)
                        break
        
        """
        actions = state.minMovesRaw()
        self.stats.addBranchFactor(len(actions))
        for i, action in enumerate(actions):
            b = state.toBoard(action)
            if b is not None:
                _, maxValue = self.maximize(b, depth-1, alpha, beta)
                if maxValue <= value:
                    best, value = action, maxValue
                    if value <= beta:
                        beta = value
                        if alpha >= beta:
                            self.stats.addPrunedBranches(len(actions) - i)
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
