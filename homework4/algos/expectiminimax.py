from algos.games import GameAlgo, GameState, NullAction, Value, Action, Player, MaxPlayer, MinPlayer
from typing import Tuple
import Grid


class ExpectiMinimax(GameAlgo):
    def maximize(self, state:GameState, depth:int) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MinValue

        actionStates = state.maxMoves()
        self.sortMoves(actionStates)
        self.stats.addBranchFactor(len(actionStates))
        for action, state in actionStates:
            _, expectedMinValue = self.minimize(state, depth-1)
            if expectedMinValue > value:
                best, value = action, expectedMinValue
        
        return best, value


    def minimize(self, state:GameState, depth:int) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MaxValue

        actionStates = state.minMoves()
        self.sortMoves(actionStates, reverse=True)
        self.stats.addBranchFactor(len(actionStates))
        for action, state in actionStates:
            _, maxValue = self.expectedValue(state, depth-1)
            if maxValue < value:
                best, value = action, maxValue
        
        return best, value


    def expectedValue(self, state:GameState, depth:int) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        expectedValue = 0
        actionStates = state.chanceMoves()
        self.stats.addBranchFactor(len(actionStates))
        for probability, state in actionStates:
            _, value = self.maximize(state, depth-1)
            expectedValue += probability * value

        return NullAction, expectedValue


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
