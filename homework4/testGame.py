import click
import GameManager
import Grid
import IntelligentAgent
import ComputerAI
import Displayer
import BaseDisplayer
import random
#import concurrent.futures
import copy
import statistics
import time
import ray

from typing import Dict, List, Iterable, Sequence, Tuple, Optional

from algos.games import CompositeEvaluator,Evaluator, GameAlgo, GameConfig, GameState, GameStatistics
from algos.evaluators import Monotonic, Snake, Corner
from algos.minimax import Minimax
from algos.alphabeta import AlphaBeta
from algos.expectiminimax import ExpectiMinimax
from algos.alphabetaExpecti import ExpectiAlphaBeta

from itertools import combinations_with_replacement, permutations

#ray.init(address='auto', _redis_password='5241590000000000')

Algorithms = {
    "abexpecti": ExpectiAlphaBeta
}

Heuristics = {
    "monotonic": Monotonic,
    "snake": Snake,
    "corner": Corner
}

DefaultTimePerMove = 0.18
DefaultAlgo = 'abexpecti'
DefaultHeuristic = 'monotonic'


class OptimizationScenario:
    def __init__(self, weights:Dict[str,int], config:GameConfig, generation:int, games:int, display:bool):
        self.weights = weights
        self.config = config
        self.generation = generation
        self.games = games
        self.display = display
        self.statistics: List[GameStatistics] = []
    
    def __str__(self):
        return f"{self.config} ({self.games} games)"
    
    def report(self):
        rep = [
            f"Weights: {self.weights}",
            f"Config: {self.config}",
            f"Games: {self.games}",
            f"MaxTile: {self.maxTile()}",
            f"MinTile: {self.minTile()}",
            f"AvgTile: {self.averageTile()}",
            f"StdDevTile: {self.stdDevTile()}",
            f"MedianTile: {self.medianTile()}",
            f"ModeTile: {self.modeTile()}"
        ]
        rep.extend([f"Game {i} Stats:\n{stat}\n" for i, stat in enumerate(self.statistics)])
        return '\n'.join(rep)

    def addStatistics(self, stats: List[GameStatistics]):
        for stat in stats:
            stat.processFinalState()
        self.statistics.extend(stats)

    def _maxTiles(self) -> List[int]:
        return [stat.maxTile for stat in self.statistics]

    def minTile(self) -> float:
        return min(self._maxTiles())

    def maxTile(self) -> float:
        return max(self._maxTiles())

    def stdDevTile(self) -> float:
        return statistics.stdev(self._maxTiles())

    def averageTile(self) -> float:
        return statistics.mean(self._maxTiles())

    def medianTile(self) -> float:
        return statistics.median(self._maxTiles())

    def modeTile(self) -> float:
        try:
            return statistics.mode(self._maxTiles())
        except:
            return 0.0
    
    def valueTile(self) -> float:
        return (self.maxTile() + self.minTile() + self.averageTile() + self.medianTile() + self.modeTile()) / 5

def printBoardAndValue(grid: Grid.Grid, heuristicName:str, heuristic: Evaluator):
    print('[')
    for i in grid.map:
        print(i)
    print(']')
    gs = GameState(grid)
    print(f"{heuristicName}: {heuristic(gs):e}")

def gridFor(inputs:List[List[int]]) -> Grid.Grid:
    grid = Grid.Grid()
    grid.map = inputs
    return grid

def getCornerGrids() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []

    grids.append(
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,8]
        ])
    )

    grids.append(
        gridFor([
            [0,0,0,8],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ])
    )

    grids.append(
        gridFor([
            [8,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ])
    )

    grids.append(
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [8,0,0,0]
        ])
    )

    return grids


def getMiddleGrids() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []
    grids.append(
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,8]
        ])
    )

    grids.append(
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,8,0],
            [0,0,0,0]
        ])
    )
    return grids

def getIslandGrids() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []
    grids.append(
        gridFor([
            [2,0,0,0],
            [2,4,8,16],
            [16,32,64,128],
            [2048,1024,512,2]
        ])
    )

    grids.append(
        gridFor([
            [2,0,0,0],
            [2,4,8,16],
            [16,32,64,128],
            [2,1024,512,2]
        ])
    )

    return grids


def getMonotonicGrids() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []
    grids.append(
        gridFor([
            [2,0,0,0],
            [2,4,8,16],
            [16,32,64,128],
            [2048,1024,512,256]
        ])
    )
    grids.append(
        gridFor([
            [2,0,0,0],
            [2,4,8,16],
            [2048,32,64,128],
            [16,1024,512,256]
        ])
    )
    grids.append(
        gridFor([
            [2,0,0,0],
            [2,4,8,16],
            [2048,32,64,128],
            [1024,1024,512,256]
        ])
    )
    grids.append(
        gridFor([
            [2,0,0,0],
            [2,4,8,16],
            [2048,32,64,128],
            [2048,512,256,0]
        ])
    )
    return grids


def getMergingGrids() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []
    grids.append(
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [128,128,16,8]
        ])
    )

    grids.append(
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [256,16,8,0]
        ])
    )
    return grids


def getAliveButNotEmpty() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []
    grids.append(
        gridFor([
            [4,4,8,16],
            [256,256,128,32],
            [2048,512,256,128],
            [2048,512,256,128]
        ])
    )
    return grids


def getDeadGrids() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []

    grids.append(
        gridFor([
            [2,4,8,16],
            [512,128,64,32],
            [1024,256,128,64],
            [2048,512,256,128]
        ])
    )
    return grids


def getCornerTestGrids() -> Dict[str, Sequence[Grid.Grid]]:
    grids: Dict[str, Sequence[Grid.Grid]] = {}
    grids['corners'] = getCornerGrids()
    grids['merging'] = getMergingGrids()
    grids['aliveButNotEmpty'] = getAliveButNotEmpty()
    grids['dead'] = getDeadGrids()
    grids['islands'] = getIslandGrids()

    return grids


def getSnakeTestGrids() -> Dict[str, Sequence[Grid.Grid]]:
    grids: Dict[str, Sequence[Grid.Grid]] = {}
    grids['corners'] = getCornerGrids()
    grids['merging'] = getMergingGrids()
    grids['aliveButNotEmpty'] = getAliveButNotEmpty()
    grids['dead'] = getDeadGrids()
    grids['islands'] = getIslandGrids()

    return grids


def getMonotonicTestGrids() -> Dict[str, Sequence[Grid.Grid]]:
    grids: Dict[str, Sequence[Grid.Grid]] = {}
    grids['corners'] = getCornerGrids()
    grids['merging'] = getMergingGrids()
    grids['aliveButNotEmpty'] = getAliveButNotEmpty()
    grids['dead'] = getDeadGrids()
    grids['islands'] = getIslandGrids()
    return grids


def getTestGrids(h: str) -> Dict[str, Sequence[Grid.Grid]]:
    grids: Dict[str, Sequence[Grid.Grid]] = {}
    grids['empty'] = [
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ])
    ]
    

    if h == 'monotonic':
        grids.update(getMonotonicTestGrids())
    elif h == 'snake':
        grids.update(getSnakeTestGrids())
    elif h == 'corner':
        grids.update(getCornerTestGrids())

    return grids


def testHeuristic(heuristicName: str):
    heuristic = Heuristics[heuristicName]()
    testGrids = getTestGrids(heuristicName)
    for testSet, grids in testGrids.items():
        print(f"================={testSet}===============")
        for grid in grids:
            printBoardAndValue(grid, heuristicName, heuristic)
            print("===================================")

def _playConfig(config: GameConfig, display:bool = False) -> GameStatistics:
    #GameManager.main()
    intelligentAgent = IntelligentAgent.IntelligentAgent(config=config)
    computerAI  = ComputerAI.ComputerAI()
    displayer   = Displayer.Displayer() if display else BaseDisplayer.BaseDisplayer()
    gameManager = GameManager.GameManager(4, intelligentAgent, computerAI, displayer)

    maxTile     = gameManager.start()
    return intelligentAgent.Statistics


def _playMultipleGames(config: GameConfig, games:int = 3, display:bool = True) -> List[GameStatistics]:
    stats = []
    for game in range(games):
        stats.append(_playConfig(config, display=display))
    return stats


@ray.remote
def _playOptimizationScenario(op: OptimizationScenario) -> OptimizationScenario:
    stats = _playMultipleGames(op.config, op.games, op.display)
    op.addStatistics(stats)
    return op 


def _crossover(parentA: Dict[str,int], parentB: Dict[str,int], probability: float) -> Sequence[Dict[str,int]]:
    if random.random() > probability:
        return []

    traitCount = random.randint(1,len(parentA)-1)
    traits: List[str] = random.choices(list(parentA.keys()), k=traitCount)

    childA = copy.deepcopy(parentA)
    childB = copy.deepcopy(parentB)
    for trait in traits:
        childA[trait], childB[trait] = childB[trait], childA[trait]

    return [childA, childB]


def _mutation(individual: Dict[str,int], probability:float, minWeight:int, maxWeight:int):
    if random.random() > probability:
        return None
    
    trait = random.choice(list(individual.keys()))
    individual[trait] = random.randint(minWeight, maxWeight)

def _evolvePopulation(individuals: List[Dict[str, int]], 
        individualReproductiveProbability: List[float], 
        crossOverProbabilty:float, 
        mutationProbability:float, 
        crossovers:int,
        minWeight:int, 
        maxWeight:int) -> Sequence[Dict[str,int]]:
    newPopulation: List[Dict[str,int]] = []
    for i in range(crossovers):
        parentA, parentB = random.choices(individuals, weights=individualReproductiveProbability, k=2)
        children = _crossover(parentA, parentB, crossOverProbabilty)
        for child in children:
            _mutation(child, mutationProbability, minWeight, maxWeight)
        newPopulation.extend(children)
    return newPopulation


def _getReproductiveProbabilityForIndividuals(population: List[OptimizationScenario]) -> List[float]:
    tileValues = [pow(op.valueTile(),2) for op in  population]
    totalValueTiles = float(sum(tileValues))
    probabilities = [tileValue / totalValueTiles for tileValue in tileValues]
    
    for op,p in zip(population, probabilities):
        print(f"{op.weights}:::: {op.averageTile()}, {op.maxTile()}, {op.minTile()}, {op.medianTile()}, {op.modeTile()}, {op.stdDevTile()}:::: {p}")
    print("......")

    return probabilities
    
def _iterateOverStateSpaceCorner():
    pStart = time.time()
    algo = Algorithms['abexpecti']
    evaluator = Heuristics['corner']
    timePerMove = 0.05
    gamesPerIteration = 3
    maxWeight = 1000
    displayGames = False
    maxFitnessGroupSize = 50

    values = list(range(0,1000, 200))
    a = { (i,j,k,l,m) for i in values
            for j in values
            for k in values
            for l in values
            for m in values 
        }

    weights = [
        {'emptyWeight': i, 'mergeableWeight': j, 'cornerWeight':k, 'totalValueWeight': l, 'unsmoothPenaltyWeight':m}
           for i,j,k,l,m in a
    ]
    
    def toOptimizationScenario(weights:Dict[str, int], generation:int, games:int, display:bool):
        return OptimizationScenario(
            weights=weights,
            config=GameConfig(algo=algo, evaluator=evaluator(**weights), timePerTurn=timePerMove),
            generation=generation,
            games = games,
            display = display)

    individualFitnesses: List[OptimizationScenario] = []
    start = time.time()
    print("......")
    print(f"Running scenarios {len(weights)}......")
    
    """run new population"""
    optimizationScenarios = [ toOptimizationScenario(w, 1, gamesPerIteration, displayGames) for w in weights ]
    futures = [ _playOptimizationScenario.remote(op) for op in optimizationScenarios]
    results = ray.get(futures)
    """add results to individualFitnesses"""
    totalCount = len(results)
    totalDone = 0
    for op in results:
        individualFitnesses.append(op)
        print(f"{op.weights}:::: {op.averageTile()}, {op.maxTile()}, {op.minTile()}, {op.medianTile()}, {op.modeTile()}, {op.stdDevTile()}")
        with open('populationResultsStateSpaceCorner.txt', 'a') as fout:
            print(op.report(), file=fout)
            print("----------------------------\n", file=fout)
            print("----------------------------\n", file=fout)
        totalDone += 1
        if totalDone % 5 == 0:
            print(f"{totalCount -  totalDone} of {totalCount} remaining...")

    individualFitnesses.sort(key=lambda op: op.averageTile(), reverse=True)
    individualFitnesses = individualFitnesses[:maxFitnessGroupSize]
    with open('bestResultsStateSpaceCorner.txt', 'a') as fout:
        for op in individualFitnesses:
            print(f"{op.weights}:::: {op.averageTile()}, {op.maxTile()}, {op.minTile()}, {op.medianTile()}, {op.modeTile()}, {op.stdDevTile()}")
            print(op.report(), file=fout)
            print("----------------------------\n", file=fout)
            print("----------------------------\n", file=fout)

    
    print(f"Start: {pStart}")
    print(f"End: {pEnd}")
    pEnd = time.time()



def _iterateOverStateSpaceMonotonicByGame():
    pStart = time.time()
    algo = Algorithms['abexpecti']
    evaluator = Heuristics['monotonic']
    timePerMove = 0.15
    gamesPerIteration = 10
    maxWeight = 1000
    displayGames = False
    maxFitnessGroupSize = 50
    
    values = list(range(0,1000, 200))
    a = { (i,j,k,l) for i in values
            for j in values
            for k in values
            for l in values }
    weights = [
        {'emptyWeight': i, 'mergeableWeight': j, 'montonicWeight': k, 'totalValueWeight': l}
           for i,j,k,l in a
    ]
    weights = [
        {'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 200},     # 4096, 4096, 4096, 4096, 4096, 0.0
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},     # 3584, 8192, 512, 2048, 2048, 4063.8740137952113
        {'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 200},     #
        {'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 400},     # 4096, 4096, 4096, 4096, 4096, 0.0
{'emptyWeight': 400, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 400},     # 4096, 8192, 2048, 2048, 2048, 3547.2400539010605
{'emptyWeight': 600, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 200},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 800},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 400},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 400},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 400},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 200},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 800, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 600},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 600},     # 3413.3333333333335, 4096, 2048, 4096, 4096, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 200},     # 3072, 4096, 1024, 4096, 4096, 1773.6200269505302
{'emptyWeight': 0, 'mergeableWeight': 0, 'montonicWeight': 400, 'totalValueWeight': 600},     # 3072, 4096, 1024, 4096, 4096, 1773.6200269505302
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 800},     # 3072, 4096, 1024, 4096, 4096, 1773.6200269505302
{'emptyWeight': 800, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 600},     # 3072, 4096, 1024, 4096, 4096, 1773.6200269505302
{'emptyWeight': 600, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 600},     # 3072, 4096, 1024, 4096, 4096, 1773.6200269505302
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 400, 'totalValueWeight': 600},     # 2901.3333333333335, 4096, 512, 4096, 4096, 2069.223364775619
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2901.3333333333335, 4096, 512, 4096, 4096, 2069.223364775619
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2901.3333333333335, 4096, 512, 4096, 4096, 2069.223364775619
{'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 600},     # 2901.3333333333335, 4096, 512, 4096, 4096, 2069.223364775619
{'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 600},     # 2816, 4096, 256, 4096, 4096, 2217.025033688163
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 200},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 800, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 400, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 0, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 0, 'montonicWeight': 400, 'totalValueWeight': 200},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 0, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2730.6666666666665, 4096, 2048, 2048, 2048, 1182.4133513003535
{'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 600, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 200},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 400},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 0, 'montonicWeight': 400, 'totalValueWeight': 200},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 200, 'totalValueWeight': 800},     # 2389.3333333333335, 4096, 1024, 2048, 0.0, 1564.1858372115935
{'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 600, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 0, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 0, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 0, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 600, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 800, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 400, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 0, 'mergeableWeight': 400, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 200, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 800, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 800},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 200, 'totalValueWeight': 600},     # 2218.6666666666665, 4096, 512, 2048, 0.0, 1798.0849071535342
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 200},     # 2133.3333333333335, 4096, 256, 2048, 0.0, 1921.4216958630745
{'emptyWeight': 0, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2133.3333333333335, 4096, 256, 2048, 0.0, 1921.4216958630745
{'emptyWeight': 200, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2133.3333333333335, 4096, 256, 2048, 0.0, 1921.4216958630745
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2090.6666666666665, 4096, 128, 2048, 0.0, 1984.3440561891814
{'emptyWeight': 600, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 200},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 600, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 800},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 0, 'mergeableWeight': 400, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 600},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 600},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 600, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 600, 'mergeableWeight': 400, 'montonicWeight': 400, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 400, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 600},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 600},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 0, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 0, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 800},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 800, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 200},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 0},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 600},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 800, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 800, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 400, 'mergeableWeight': 0, 'montonicWeight': 600, 'totalValueWeight': 400},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 0, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 400, 'totalValueWeight': 800},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 0, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 200},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 200},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 200},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 200, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 200},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 800},     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
{'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 800},     # 2048, 2048, 2048, 2048, 2048, 0.0
{'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 200, 'totalValueWeight': 400}     # 2048, 4096, 1024, 1024, 1024, 1773.6200269505302
    ]

    def toOptimizationScenario(weights:Dict[str, int], generation:int, games:int, display:bool):
        return OptimizationScenario(
            weights=weights,
            config=GameConfig(algo=algo, evaluator=evaluator(**weights), timePerTurn=timePerMove),
            generation=generation,
            games = games,
            display = display)

    start = time.time()
    print("......")
    
    scenarios: List[OptimizationScenario] = [ toOptimizationScenario(w, 1, 1, displayGames) for w in weights ]
    """run new population"""
    optimizationScenarios: List[OptimizationScenario] = [ toOptimizationScenario(w, 1, 1, displayGames) for w in weights
                    for i in range(gamesPerIteration) ]
    print(f"Running scenarios {len(weights)}......")
    futures = [ _playOptimizationScenario.remote(op) for op in optimizationScenarios]

    totalCount = len(futures)
    totalDone = 0
    while len(futures):
        done, futures = ray.wait(futures)
        totalDone += len(done)
        for d in done:
            result = ray.get(d)
            for scenario in scenarios:
                if result.weights == scenario.weights:
                    scenario.addStatistics(result.statistics)
        if totalDone % 10 == 0:
            print(f"Finished {totalDone} of {totalCount}")

    scenarios.sort(key=lambda op: op.averageTile(), reverse=True)
    with open('bestResultsStateSpace2.txt', 'a') as fout:
        for op in scenarios:
            print(f"{op.weights}:::: {op.averageTile()}, {op.maxTile()}, {op.minTile()}, {op.medianTile()}, {op.modeTile()}, {op.stdDevTile()}")
            print(op.report(), file=fout)
            print("----------------------------\n", file=fout)
            print("----------------------------\n", file=fout)

    
    pEnd = time.time()
    print(f"Start: {pStart}")
    print(f"End: {pEnd}")



def _iterateOverStateSpaceMonotonic():
    pStart = time.time()
    algo = Algorithms['abexpecti']
    evaluator = Heuristics['monotonic']
    timePerMove = 0.18
    gamesPerIteration = 10
    maxWeight = 1000
    displayGames = False
    maxFitnessGroupSize = 50
    """
    values = list(range(0,1000, 200))
    a = { (i,j,k,l) for i in values
            for j in values
            for k in values
            for l in values }
    weights = [
        {'emptyWeight': i, 'mergeableWeight': j, 'montonicWeight': k, 'totalValueWeight': l}
           for i,j,k,l in a
    ]
    """
    weights = [
        {'emptyWeight': 200, 'mergeableWeight': 0, 'montonicWeight': 200, 'totalValueWeight': 200},     # 4096, 4096, 4096, 4096, 4096, 0.0
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},     # 3584, 8192, 512, 2048, 2048, 4063.8740137952113
        {'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 200}     #
    ]
    
    def toOptimizationScenario(weights:Dict[str, int], generation:int, games:int, display:bool):
        return OptimizationScenario(
            weights=weights,
            config=GameConfig(algo=algo, evaluator=evaluator(**weights), timePerTurn=timePerMove),
            generation=generation,
            games = games,
            display = display)

    individualFitnesses: List[OptimizationScenario] = []
    start = time.time()
    print("......")
    print(f"Running scenarios {len(weights)}......")
    
    """run new population"""
    optimizationScenarios = [ toOptimizationScenario(w, 1, gamesPerIteration, displayGames) for w in weights ]
    futures = [ _playOptimizationScenario.remote(op) for op in optimizationScenarios]
    results = ray.get(futures)

    """add results to individualFitnesses"""
    totalCount = len(results)
    totalDone = 0
    for op in results:
        individualFitnesses.append(op)
        print(f"{op.weights}:::: {op.averageTile()}, {op.maxTile()}, {op.minTile()}, {op.medianTile()}, {op.modeTile()}, {op.stdDevTile()}")
        with open('populationResultsStateSpace2.txt', 'a') as fout:
            print(op.report(), file=fout)
            print("----------------------------\n", file=fout)
            print("----------------------------\n", file=fout)
        totalDone += 1
        if totalDone % 5 == 0:
            print(f"{totalCount -  totalDone} of {totalCount} remaining...")

    individualFitnesses.sort(key=lambda op: op.averageTile(), reverse=True)
    individualFitnesses = individualFitnesses[:maxFitnessGroupSize]
    with open('bestResultsStateSpace2.txt', 'a') as fout:
        for op in individualFitnesses:
            print(f"{op.weights}:::: {op.averageTile()}, {op.maxTile()}, {op.minTile()}, {op.medianTile()}, {op.modeTile()}, {op.stdDevTile()}")
            print(op.report(), file=fout)
            print("----------------------------\n", file=fout)
            print("----------------------------\n", file=fout)

    
    print(f"Start: {pStart}")
    print(f"End: {pEnd}")
    pEnd = time.time()

def _optimizeMonotonic():
    pStart = time.time()
    algo = Algorithms['abexpecti']
    evaluator = Heuristics['monotonic']
    timePerMove = 0.05
    gamesPerIteration = 5
    crossoverCount = 10
    initialPopulationSize = 50  #start with a very large initial population
    maxFitnessGroupSize = 30
    maxIter = 25
    crossOverProbabilty = 0.9
    mutationProbability = 0.3
    minWeight = 0
    maxWeight = 1000
    displayGames = False
    population = [
        {'emptyWeight': 82, 'mergeableWeight': 220, 'montonicWeight': 821, 'totalValueWeight': 713},
        {'emptyWeight': 655, 'mergeableWeight': 516, 'montonicWeight': 821, 'totalValueWeight': 919},
        {'emptyWeight': 578, 'mergeableWeight': 419, 'montonicWeight': 629, 'totalValueWeight': 911},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 629, 'totalValueWeight': 919},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 343, 'totalValueWeight': 919},
        {'emptyWeight': 655, 'mergeableWeight': 872, 'montonicWeight': 821, 'totalValueWeight': 919},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 629, 'totalValueWeight': 911},
        {'emptyWeight': 578, 'mergeableWeight': 516, 'montonicWeight': 821, 'totalValueWeight': 919},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 629, 'totalValueWeight': 911},
        {'emptyWeight': 151, 'mergeableWeight': 716, 'montonicWeight': 629, 'totalValueWeight': 286},
        {'emptyWeight': 421, 'mergeableWeight': 419, 'montonicWeight': 629, 'totalValueWeight': 919},
        {'emptyWeight': 578, 'mergeableWeight': 716, 'montonicWeight': 200, 'totalValueWeight': 143},
        {'emptyWeight': 421, 'mergeableWeight': 516, 'montonicWeight': 821, 'totalValueWeight': 911},
        {'emptyWeight': 655, 'mergeableWeight': 872, 'montonicWeight': 629, 'totalValueWeight': 919},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 821, 'totalValueWeight': 911}
    ]
    population.extend([
        {'emptyWeight':random.randint(minWeight, maxWeight),
         'mergeableWeight':random.randint(minWeight, maxWeight),
         'montonicWeight':random.randint(minWeight, maxWeight),
         'totalValueWeight':random.randint(minWeight, maxWeight) }
         for i in range(initialPopulationSize)
    ])
    
    with open('populationEvolution.txt', 'w') as fout:
        print("Population evolutions\n\n",file=fout)
    
    def toOptimizationScenario(weights:Dict[str, int], generation:int, games:int, display:bool):
        return OptimizationScenario(
            weights=weights,
            config=GameConfig(algo=algo, evaluator=evaluator(**weights), timePerTurn=timePerMove),
            generation=generation,
            games = games,
            display = display)

    individualFitnesses: List[OptimizationScenario] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for generation in range(maxIter):
            start = time.time()
            print(f"----------------------------")
            print(f"----------------------------")
            print(f"Generation {generation} (Population Size: {len(population)})...")

            """record new population"""
            with open('populationEvolution.txt', 'a') as fout:
                for p in population:
                    print(p, file=fout)
                    print(p)

            print("......")
            print("Running scenarios......")
            
            """run new population"""
            optimizationScenarios = [ toOptimizationScenario(w, generation, gamesPerIteration, displayGames) for w in population ]
            futures = [ executor.submit(_playOptimizationScenario, op) for op in optimizationScenarios ]
            """add results to individualFitnesses"""
            totalCount = len(futures)
            totalDone = 0
            for future in concurrent.futures.as_completed(futures):
                individualFitnesses.append(future.result())
                totalDone += 1
                if totalDone % 5 == 0:
                    print(f"{totalCount -  totalDone} of {totalCount} remaining...")

            
            """take top [maxFitnessGroupSize] fitnesses, write to file"""
            print("Recording results......")
            individualFitnesses.sort(key=lambda op: op.averageTile(), reverse=True)
            individualFitnesses = individualFitnesses[:maxFitnessGroupSize]
            with open('populationResults.txt', 'w') as fout:
                print("Population results\n\n", file=fout)
                for op in individualFitnesses:
                    print(op.report(), file=fout)
                    print("----------------------------\n", file=fout)
                    print("----------------------------\n", file=fout)
            
            """get new population with individuals with top fitnesses"""
            print("Evolving population......")
            individuals = [op.weights for op in individualFitnesses]
            reproductiveProbability = _getReproductiveProbabilityForIndividuals(individualFitnesses)
            population = _evolvePopulation(
                individuals=individuals,
                individualReproductiveProbability=reproductiveProbability,
                crossOverProbabilty=crossOverProbabilty,
                mutationProbability=mutationProbability,
                crossovers=crossoverCount,
                minWeight=minWeight,
                maxWeight=maxWeight)
            end = time.time()

            print(f"Started: {start}")
            print(f"Ended: {end}")
            print(f"Total time: {end - start}")
    
    pEnd = time.time()
    print("Genetic Algorithm stats:")
    print(f"Iterations: {maxIter}")
    print(f"Start: {pStart}")
    print(f"End: {pEnd}")


def _playGameAndReport(algorithm:str, heuristicName:str, timePerMove:float):
    algo = Algorithms[algorithm]
    evaluator = Heuristics[heuristicName]()
    config = GameConfig(algo=algo, evaluator=evaluator, timePerTurn=timePerMove)
    stats = _playConfig(config, display=True)
    print(f"--Config--\n{config}\n")
    print(f"--Statistics--\n{stats}\n")


@click.group()
def run():
    pass


@run.command()
def optimizecorner():
    _iterateOverStateSpaceCorner()

@run.command()
def optimizemonotonic():
    _iterateOverStateSpaceMonotonicByGame()


@run.command()
@click.option('--heuristic', '-h',
    default=DefaultHeuristic,
    type=click.Choice(list(Heuristics), case_sensitive=True))
def checkheuristic(heuristic):
    testHeuristic(heuristic)

@run.command()
@click.option('--algorithm', '-a', 
    default=DefaultAlgo,
    type=click.Choice(list(Algorithms), case_sensitive=True))
@click.option('--heuristic', '-h',
    default=DefaultHeuristic,
    type=click.Choice(list(Heuristics), case_sensitive=True))
@click.option('--timepermove', '-t', default=DefaultTimePerMove)
def playgame(algorithm, heuristic, timepermove):
    print(f'Algorithm: {algorithm}')
    print(f'Heuristics: {heuristic}')
    print(f'Time per move: {timepermove}')
    _playGameAndReport(algorithm, heuristic, timepermove)

if __name__ == "__main__":
    run()