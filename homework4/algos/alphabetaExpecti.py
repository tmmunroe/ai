from algos.games import GameAlgo, GameState, NullAction, Value, Action
from typing import Any, Tuple, Dict, Sequence
import Grid
import sys

class ExpectiAlphaBeta(GameAlgo):
    def maximize(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MinValue
        actionStates = state.maxMoves()
        self.sortMoves(actionStates, reverse=True)
        #print(f"{state.prefix}Max MoveOrder: {[self.evaluate(a) for _,a in actionStates]}")
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, b) in enumerate(actionStates):
            _, minValue = self.expectedValue(b, depth-1, alpha, beta)
            #print(f"{state.prefix}Max Action: {action}, ExpectedValue: {minValue}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}, Best: {best}, BestValue: {value}")
            if minValue >= value:
                best, value = action, minValue
                if value >= alpha:
                    #print(f"{state.prefix}Setting Alpha:{value} from {alpha}, Depth:{depth}")
                    alpha = value
                    if alpha >= beta:
                        #print(f"{state.prefix}Pruning at Max Alpha:{alpha}, Beta:{beta}, Depth:{depth}")
                        self.stats.addPrunedBranches(len(actionStates) - i)
                        break
        
        return best, value

    def expectedValue(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        expectedValue = 0
        chanceState = state.chanceState()
        chanceMoves = chanceState.chanceMoves()
        self.stats.addBranchFactor(len(chanceMoves))
        for probability, newTile in chanceMoves:
            _, value = self.minimize(chanceState, newTile, depth-1, alpha, beta)
            #print(f"{state.prefix}ExpectedValue: Tile: {newTile}, Value: {value}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}")
            expectedValue += probability * value

        return NullAction, expectedValue

    def minimize(self, state:GameState, newTile:int, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MaxValue

        actionStates = state.minMoves(newTile)
        self.sortMoves(actionStates, reverse=False)
        #print(f"{state.prefix}Min MoveOrder: {[self.evaluate(a) for _,a in actionStates]}")
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, b) in enumerate(actionStates):
            _, maxValue = self.maximize(b, depth-1, alpha, beta)
            #print(f"{state.prefix}Min Action: {action}, ExpectedValue: {maxValue}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}, Best: {best}, BestValue: {value}")
            if maxValue <= value:
                best, value = action, maxValue
                if value <= beta:
                    #print(f"{state.prefix}Setting Beta:{value} from {beta}, Depth:{depth}")
                    beta = value
                    if alpha >= beta:
                        #print(f"{state.prefix}Pruning at Min Alpha:{alpha}, Beta:{beta}, Depth:{depth}")
                        self.stats.addPrunedBranches(len(actionStates) - i)
                        break
        return best, value


    def search(self) -> Action:
        depth = 1
        alpha = float('-inf')
        beta = float('inf')
        while True:
            #print(f"Searching to depth {depth}")
            self.initialState.prefix = "    " * depth
            best, value = self.maximize(self.initialState, depth, alpha, beta)
            self.checkAndSetAction(best, value)
            if self.searchTerminated():
                break
            depth += 1
            #if depth == 7:
            #    sys.exit()
            #print()
        self.stats.addSearchDepth(depth)
        return self.bestAction()
