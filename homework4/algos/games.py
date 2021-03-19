import random
import threading
import time
from typing import Any, Callable, Iterable, NewType, Protocol, Sequence, Tuple, Optional, Union, Type, List, Dict
import Grid
import random

Value = float
Action = int
NullAction = Action(-1)
Player = int

MinPlayer, MaxPlayer = Player(0), Player(1)

HeuristicCache:Dict[int,Value] = {}

class GameState:
    def __init__(self, state: Grid.Grid, newTile: int = 0):
        self.state = state
        self.newTile = newTile
        self.gridHash = None
        if self.newTile == 0:
            self.newTile = 2 if random.uniform(0,1) < 0.9 else 4
    
    def hashGrid(self):
        if self.gridHash is None:
            self.gridHash = hash(tuple(tuple(r) for r in self.state.map))
        return self.gridHash

    def maxMoves(self):
        actionStates = self.state.getAvailableMoves()
        return [ (a, GameState(grid)) for a,grid in actionStates ]

    def minMoves(self):
        actionStates = []
        availableCells = self.state.getAvailableCells()
        for cell in availableCells:
            if cell and self.state.canInsert(cell):
                gridCopy = self.state.clone()
                gridCopy.setCellValue(cell, self.newTile)
                actionStates.append((cell, GameState(gridCopy)))
        return actionStates

    def chanceMoves(self):
        twoTile = 0.9, GameState(self.state.clone(), 2)
        fourTile = 0.1, GameState(self.state.clone(), 4)
        return [ twoTile, fourTile ]


class Evaluator:
    def __call__(self, state:GameState) -> Value:
        return random.uniform(self.MinValue, self.MaxValue)

    @property
    def MinValue(self) -> Value:
        return 0

    @property
    def MaxValue(self) -> Value:
        return 1


class CompositeEvaluator(Evaluator):
    def __init__(self, heuristics: Sequence[Evaluator], weights: Sequence[float]):
        self.heuristics = [(h,w) for h,w in zip(heuristics, weights)]
        self.minValue = sum((w*h.MinValue for h,w in self.heuristics))
        self.maxValue = sum((w*h.MaxValue for h,w in self.heuristics))

    def __call__(self, state:GameState) -> Value:
        return sum((w*h(state) for h,w in self.heuristics))

    @property
    def MinValue(self) -> Value:
        return self.minValue

    @property
    def MaxValue(self) -> Value:
        return self.maxValue

class GameStatistics:
    def __init__(self):
        self.finalState: GameState = None
        self.maxTile: int = 0
        self.sumOfTiles: int = 0
        self.averageTile: float = 0

        self.searches: int = 0
        self.sumOfDepths: int = 0
        self.maxDepth: int = 0
        self.minDepth: int = 0

        self.totalNodes: int = 0
        self.sumOfBranches: int = 0
        self.maxBranches: int = 0
        self.minBranches: int = 0

        self.prunings: int = 0
        self.sumOfPrunings: int = 0
    
    def addSearchDepth(self, depth:int):
        self.searches += 1
        self.sumOfDepths += depth
        self.maxDepth = max(self.maxDepth, depth)
        self.minDepth = min(self.minDepth, depth)
    
    def addBranchFactor(self, branches:int):
        self.totalNodes += 1
        self.sumOfBranches += branches
        self.maxBranches = max(self.maxBranches, branches)
        self.minBranches = min(self.minBranches, branches)

    def addPrunedBranches(self, pruned:int):
        self.prunings += 1
        self.sumOfPrunings += pruned
    
    def updateState(self, state: GameState):
        self.finalState = state

    def processFinalState(self):
        self.maxTile = self.finalState.state.getMaxTile()
        self.sumOfTiles = sum(sum(row) for row in self.finalState.state.map)
        self.averageTile = self.sumOfTiles / (self.finalState.state.size*self.finalState.state.size)

    def __str__(self):
        self.processFinalState()
        return '\n'.join((f"Searches: {self.searches}",
            f"TotalDepth:{self.sumOfDepths}",
            f"AverageDepth:{float(self.sumOfDepths)/self.searches}",
            f"MaxDepth:{self.maxDepth}",
            f"MinDepth:{self.minDepth}",
            f"Nodes:{self.totalNodes}",
            f"TotalBranches:{self.sumOfBranches}",
            f"AverageBranches:{float(self.sumOfBranches)/self.totalNodes}",
            f"MaxBranches:{self.maxBranches}",
            f"MinBranches:{self.minBranches}",
            f"Prunings:{self.prunings}",
            f"SumOfPrunings:{self.sumOfPrunings}",
            f"AveragedPrunings:{self.sumOfPrunings/self.prunings if self.prunings != 0 else 0}",
            f"MaxTile:{self.maxTile}",
            f"SumOfTiles:{self.sumOfTiles}",
            f"AverageTile:{self.averageTile}"
        ))


class GameAlgo:
    def __init__(self, state: GameState, evaluator: Evaluator, stats:GameStatistics):
        self.initialState = state
        self.evaluator = evaluator
        self.bestMove: Action = NullAction
        self.bestMoveValue: Value = evaluator.MinValue
        self.moveLock = threading.Lock()
        self.stop = threading.Event()
        self.stats = stats

    def sortMoves(self, moves:List[Tuple[Any,GameState]], reverse=False):
        moves.sort(key=lambda tup: self.evaluate(tup[1]), reverse=reverse)

    def search(self) -> Action:
        self.stats.addSearchDepth(1)
        moveset = self.initialState.maxMoves()
        self.stats.addBranchFactor(len(moveset))
        return random.choice(moveset)[0] if moveset else NullAction

    def evaluate(self, state: GameState) -> float:
        global HeuristicCache
        if state.hashGrid() not in HeuristicCache:
            h = self.evaluator(state)
            HeuristicCache[state.hashGrid()] = h
            return h
        return HeuristicCache[state.hashGrid()]

    def terminateSearch(self) -> Action:
        #global SubTreeCache
        #SubTreeCache.clear()
        self.stop.set()
        return self.bestAction()
        
    def searchTerminated(self) -> bool:
        return self.stop.is_set()

    def bestAction(self) -> Action:
        with self.moveLock:
            return self.bestMove
    
    def checkAndSetAction(self, action:Action, value:Value) -> None:
        with self.moveLock:
            if value > self.bestMoveValue:
                self.bestMove, self.bestMoveValue = action, value

class GameConfig:
    def __init__(self, algo: Type[GameAlgo] = GameAlgo, 
                    evaluator: Evaluator = Evaluator(), 
                    timePerTurn: float = 0.2):
        self.Algo = algo
        self.Evaluator = evaluator
        self.TimePerTurn = timePerTurn
