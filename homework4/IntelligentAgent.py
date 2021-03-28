from BaseAI import BaseAI
import concurrent.futures
import threading
from typing import Any, Dict, List, Tuple, Type
import Grid
import random
import sys

PrintDebuggingOutput = False

Value = float
Action = int
NullAction = Action(-1)

ChanceMoves = (
    (0.9, 2),
    (0.1, 4)
)

DefaultAlgo = ExpectiAlphaBeta
DefaultEvaluator = Monotonic
DefaultTimePerMove = 0.18

def getGameConfig():
    return GameConfig(algo=DefaultAlgo, evaluator=DefaultEvaluator(), timePerTurn=DefaultTimePerMove)


class GameState:
    def __init__(self, state: Grid.Grid, prefix:str = ''):
        self.state = state
        self.gridHash = None
        self.prefix = prefix
    
    def hashGrid(self):
        if self.gridHash is None:
            self.gridHash = hash(tuple(tuple(r) for r in self.state.map))
        return self.gridHash

    def maxMoves(self):
        actionStates = self.state.getAvailableMoves()
        return [ (a, GameState(grid, prefix=self.prefix[:-4])) for a,grid in actionStates ]

    def chanceState(self):
        gridCopy = self.state.clone()
        return GameState(gridCopy, prefix=self.prefix[:-4])

    def minMoves(self, newTile):
        actionStates = []
        availableCells = self.state.getAvailableCells()
        for cell in availableCells:
            #print(f"Adding new tile: {newTile}")
            gridCopy = self.state.clone()
            gridCopy.setCellValue(cell, newTile)
            actionStates.append((cell, GameState(gridCopy, prefix=self.prefix[:-4])))
        return actionStates
    
    def chanceMoves(self):
        return ChanceMoves


class Evaluator:
    def __call__(self, state:GameState) -> Value:
        return random.uniform(self.MinValue, self.MaxValue)

    def __str__(self):
        return "Evaluator"

    @property
    def MinValue(self) -> Value:
        return 0

    @property
    def MaxValue(self) -> Value:
        return 1


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

        self.maxCacheSize = 0
        self.cacheFlushes = 0
        self.cacheHits = 0
        self.cacheMisses = 0
    
    def updateCacheStats(self, cache: Dict[int,Value]):
        self.maxCacheSize = max(self.maxCacheSize, len(cache))

    def flushedCache(self):
        self.cacheFlushes += 1

    def addCacheHit(self):
        self.cacheHits += 1

    def addCacheMiss(self):
        self.cacheMisses += 1
    
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
        return '\n'.join((
            f"MaxTile:{self.maxTile}",
            f"Searches: {self.searches}",
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
            f"SumOfTiles:{self.sumOfTiles}",
            f"AverageTile:{self.averageTile}",
            f"MaxCacheSize: {self.maxCacheSize}",
            f"CacheFlushed: {self.cacheFlushes}",
            f"CacheHits:{self.cacheHits}",
            f"CacheMisses:{self.cacheMisses}"
        ))


class GameAlgo:
    def __init__(self, state: GameState, evaluator: Evaluator, stats:GameStatistics, heuristicCache:Dict[int,Value]):
        self.initialState = state
        self.evaluator = evaluator
        #self.bestMove: Action = state.state.getAvailableMoves()[0]
        self.bestMove = NullAction
        self.bestMoveValue: Value = evaluator.MinValue
        self.moveLock = threading.Lock()
        self.stop = threading.Event()
        self.stats = stats
        self.heuristicCache = heuristicCache

    def sortMoves(self, moves:List[Tuple[Any,GameState]], reverse=False):
        moves.sort(key=lambda tup: self.evaluate(tup[1]), reverse=reverse)

    def search(self) -> Action:
        self.stats.addSearchDepth(1)
        moveset = self.initialState.maxMoves()
        self.stats.addBranchFactor(len(moveset))
        return random.choice(moveset)[0] if moveset else NullAction

    def evaluate(self, state: GameState) -> float:
        hg = state.hashGrid()
        if hg in self.heuristicCache:
            self.stats.addCacheHit()
            return self.heuristicCache[hg]

        h = self.evaluator(state)
        self.heuristicCache[hg] = h
        self.stats.addCacheMiss()
        return h

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
    def __init__(self, algo: Type[GameAlgo], 
                    evaluator: Evaluator, 
                    timePerTurn: float):
        self.Algo = algo
        self.Evaluator = evaluator
        self.TimePerTurn = timePerTurn

    def __str__(self):
        return '\n'.join((f"Algo: {self.Algo}",
            f"Evaluator: {self.Evaluator}",
            f"TimePerTurn: {self.TimePerTurn}"
        ))


class Monotonic(Evaluator):
    """primarily inspired by stackoverflow article referenced in homework, particularly the approach to
        applying monotonicity as a penalty that scales with the size of the tiles involved
       other than weighting, the main difference is that I add the total value measure to the score
         rather than subtracting it
       return -inf if the board is unplayable
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


class ExpectiAlphaBeta(GameAlgo):
    def maximize(self, state:GameState, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MinValue
        actionStates = state.maxMoves()
        self.sortMoves(actionStates, reverse=True)
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, b) in enumerate(actionStates):
            _, minValue = self.expectedValue(b, depth-1, alpha, beta)
            if PrintDebuggingOutput:
                print(f"{state.prefix}Max Action: {action}, ExpectedValue: {minValue}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}, Best: {best}, BestValue: {value}")
            if minValue >= value:
                best, value = action, minValue
                if value >= alpha:
                    #print(f"{state.prefix}Setting Alpha:{value} from {alpha}, Depth:{depth}")
                    alpha = value
                    if alpha >= beta:
                        if PrintDebuggingOutput:
                            print(f"{state.prefix}Pruning at Max Alpha:{alpha}, Beta:{beta}, Depth:{depth}")
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
            if PrintDebuggingOutput:
                print(f"{state.prefix}ExpectedValue: Tile: {newTile}, Value: {value}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}")
            expectedValue += probability * value

        return NullAction, expectedValue

    def minimize(self, state:GameState, newTile:int, depth:int, alpha:Value, beta:Value) -> Tuple[Action, Value]:
        if depth == 0 or self.searchTerminated():
            return NullAction, self.evaluate(state)
        
        best:Action = NullAction
        value:Value = self.evaluator.MaxValue

        actionStates = state.minMoves(newTile)
        self.sortMoves(actionStates, reverse=False)
        self.stats.addBranchFactor(len(actionStates))
        for i, (action, b) in enumerate(actionStates):
            _, maxValue = self.maximize(b, depth-1, alpha, beta)
            if PrintDebuggingOutput:
                print(f"{state.prefix}Min Action: {action}, ExpectedValue: {maxValue}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}, Best: {best}, BestValue: {value}")
            if maxValue <= value:
                best, value = action, maxValue
                if value <= beta:
                    #print(f"{state.prefix}Setting Beta:{value} from {beta}, Depth:{depth}")
                    beta = value
                    if alpha >= beta:
                        if PrintDebuggingOutput:
                            print(f"{state.prefix}Pruning at Min Alpha:{alpha}, Beta:{beta}, Depth:{depth}")
                        self.stats.addPrunedBranches(len(actionStates) - i)
                        break
        return best, value


    def search(self) -> Action:
        depth = 1
        alpha = float('-inf')
        beta = float('inf')
        while True:
            if PrintDebuggingOutput:
                print(f"Searching to depth {depth}")
            self.initialState.prefix = "    " * depth
            best, value = self.maximize(self.initialState, depth, alpha, beta)
            self.checkAndSetAction(best, value)
            if self.searchTerminated():
                break
            depth += 1
            if PrintDebuggingOutput:
                if depth == 4:
                    sys.exit()
                print()
        self.stats.addSearchDepth(depth)
        return self.bestAction()


class IntelligentAgent(BaseAI):
	def __init__(self, config = getGameConfig()):
		self.Executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
		self.Config: GameConfig = config
		self.Statistics: GameStatistics = GameStatistics()
		self.HeuristicCache:Dict[int,Value] = {}
		self.MaxTileSeen = 0

	def getMove(self, grid):
		gridState = GameState(grid)
		maxTile = grid.getMaxTile()
		
		if maxTile > self.MaxTileSeen:
			self.HeuristicCache.clear()
			self.Statistics.flushedCache()
			self.MaxTileSeen = maxTile
		
		self.Statistics.updateState(gridState)
		algo = self.Config.Algo(gridState, self.Config.Evaluator, self.Statistics, self.HeuristicCache)
		s = self.Executor.submit(algo.search)
		a = NullAction
		try:
			a = s.result(timeout=self.Config.TimePerTurn)
		except concurrent.futures.TimeoutError:
			a = algo.terminateSearch()

		self.Statistics.updateCacheStats(self.HeuristicCache)
		return a if a != NullAction else None