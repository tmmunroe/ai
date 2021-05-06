import click
import GameManager
import Grid
from IntelligentAgent import Evaluator, GameAlgo, GameConfig, GameState, GameStatistics, Monotonic, ExpectiAlphaBeta, IntelligentAgent, getGameConfig
import ComputerAI
import Displayer
import BaseDisplayer
import random
#import concurrent.futures
import copy
import statistics
import time
import ray
import multiprocessing

from typing import Dict, List, Iterable, Sequence, Tuple, Optional

ray.init(address='auto', _redis_password='5241590000000000')

Algorithms = {
    "abexpecti": ExpectiAlphaBeta
}

Heuristics = {
    "monotonic": Monotonic
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
    
    def report(self, includeGames=False):
        rep = [
            f"Weights: {self.weights}",
            f"Config: {self.config}",
            f"Games: {self.games}",
            f"MaxTile: {self.maxTile()}",
            f"MinTile: {self.minTile()}",
            f"AvgTile: {self.averageTile()}",
            f"StdDevTile: {self.stdDevTile()}",
            f"MedianTile: {self.medianTile()}",
            f"ModeTile: {self.modeTile()}",
            f"Game max tiles: { sorted(self._maxTiles(), reverse=True) }"
        ]
        if includeGames:
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
        return (self.maxTile() + self.minTile() + self.averageTile() + self.medianTile()) / 5

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
    intelligentAgent = IntelligentAgent(config=config)
    computerAI  = ComputerAI.ComputerAI()
    displayer   = Displayer.Displayer() if display else BaseDisplayer.BaseDisplayer()
    gameManager = GameManager.GameManager(4, intelligentAgent, computerAI, displayer)

    maxTile     = gameManager.start()
    return intelligentAgent.Statistics

def _playGame(display:bool = False) -> GameStatistics:
    #GameManager.main()
    intelligentAgent = IntelligentAgent()
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
    minWeight = 1
    maxWeight = 1000
    randomSamples = 300
    displayGames = False
    maxFitnessGroupSize = 50
    
    """
    weights = [
        {'emptyWeight':random.randint(minWeight, maxWeight),
         'mergeableWeight':random.randint(minWeight, maxWeight),
         'montonicWeight':random.randint(minWeight, maxWeight),
         'totalValueWeight':random.randint(minWeight, maxWeight) }
         for i in range(randomSamples)
    ]
    """
    weights = [
        {'emptyWeight': 695, 'mergeableWeight': 812, 'montonicWeight': 727, 'totalValueWeight': 617},
        {'emptyWeight': 60, 'mergeableWeight': 679, 'montonicWeight': 942, 'totalValueWeight': 716},
        {'emptyWeight': 172, 'mergeableWeight': 294, 'montonicWeight': 245, 'totalValueWeight': 410},
        {'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 600},
        {'emptyWeight': 430, 'mergeableWeight': 829, 'montonicWeight': 895, 'totalValueWeight': 775},
        {'emptyWeight': 49, 'mergeableWeight': 566, 'montonicWeight': 723, 'totalValueWeight': 506},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 821, 'totalValueWeight': 911},
        {'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 400},
        {'emptyWeight': 210, 'mergeableWeight': 834, 'montonicWeight': 209, 'totalValueWeight': 763},
        {'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 865, 'mergeableWeight': 886, 'montonicWeight': 740, 'totalValueWeight': 515},
        {'emptyWeight': 713, 'mergeableWeight': 733, 'montonicWeight': 235, 'totalValueWeight': 999},
        {'emptyWeight': 922, 'mergeableWeight': 176, 'montonicWeight': 336, 'totalValueWeight': 194},
        {'emptyWeight': 107, 'mergeableWeight': 448, 'montonicWeight': 233, 'totalValueWeight': 365},
        {'emptyWeight': 406, 'mergeableWeight': 712, 'montonicWeight': 945, 'totalValueWeight': 628},
        {'emptyWeight': 134, 'mergeableWeight': 535, 'montonicWeight': 463, 'totalValueWeight': 776},
        {'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 828, 'mergeableWeight': 56, 'montonicWeight': 938, 'totalValueWeight': 660},
        {'emptyWeight': 124, 'mergeableWeight': 637, 'montonicWeight': 593, 'totalValueWeight': 236},
        {'emptyWeight': 387, 'mergeableWeight': 788, 'montonicWeight': 163, 'totalValueWeight': 528},
        {'emptyWeight': 445, 'mergeableWeight': 258, 'montonicWeight': 813, 'totalValueWeight': 736},
        {'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 200},
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 400},
        {'emptyWeight': 691, 'mergeableWeight': 874, 'montonicWeight': 202, 'totalValueWeight': 241},
        {'emptyWeight': 631, 'mergeableWeight': 79, 'montonicWeight': 430, 'totalValueWeight': 776},
        {'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 600},
        {'emptyWeight': 718, 'mergeableWeight': 71, 'montonicWeight': 390, 'totalValueWeight': 669},
        {'emptyWeight': 434, 'mergeableWeight': 290, 'montonicWeight': 446, 'totalValueWeight': 678},
        {'emptyWeight': 533, 'mergeableWeight': 266, 'montonicWeight': 801, 'totalValueWeight': 824},
        {'emptyWeight': 723, 'mergeableWeight': 17, 'montonicWeight': 777, 'totalValueWeight': 321},
        {'emptyWeight': 601, 'mergeableWeight': 672, 'montonicWeight': 697, 'totalValueWeight': 571},
        {'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 800},
        {'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},
        {'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 400},
        {'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 800},
        {'emptyWeight': 565, 'mergeableWeight': 376, 'montonicWeight': 408, 'totalValueWeight': 679},
        {'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 400},
        {'emptyWeight': 368, 'mergeableWeight': 536, 'montonicWeight': 805, 'totalValueWeight': 204},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 629, 'totalValueWeight': 911},
        {'emptyWeight': 5, 'mergeableWeight': 396, 'montonicWeight': 71, 'totalValueWeight': 75},
        {'emptyWeight': 367, 'mergeableWeight': 582, 'montonicWeight': 157, 'totalValueWeight': 347},
        {'emptyWeight': 39, 'mergeableWeight': 414, 'montonicWeight': 394, 'totalValueWeight': 147},
        {'emptyWeight': 468, 'mergeableWeight': 545, 'montonicWeight': 486, 'totalValueWeight': 264},
        {'emptyWeight': 405, 'mergeableWeight': 603, 'montonicWeight': 996, 'totalValueWeight': 652},
        {'emptyWeight': 362, 'mergeableWeight': 349, 'montonicWeight': 205, 'totalValueWeight': 799},
        {'emptyWeight': 481, 'mergeableWeight': 367, 'montonicWeight': 583, 'totalValueWeight': 567},
        {'emptyWeight': 504, 'mergeableWeight': 953, 'montonicWeight': 277, 'totalValueWeight': 766},
        {'emptyWeight': 375, 'mergeableWeight': 388, 'montonicWeight': 405, 'totalValueWeight': 731},
        {'emptyWeight': 23, 'mergeableWeight': 195, 'montonicWeight': 946, 'totalValueWeight': 531},
        {'emptyWeight': 908, 'mergeableWeight': 472, 'montonicWeight': 524, 'totalValueWeight': 574},
        {'emptyWeight': 604, 'mergeableWeight': 28, 'montonicWeight': 243, 'totalValueWeight': 546},
        {'emptyWeight': 776, 'mergeableWeight': 916, 'montonicWeight': 722, 'totalValueWeight': 574},
        {'emptyWeight': 669, 'mergeableWeight': 4, 'montonicWeight': 865, 'totalValueWeight': 465},
        {'emptyWeight': 742, 'mergeableWeight': 729, 'montonicWeight': 838, 'totalValueWeight': 589},
        {'emptyWeight': 227, 'mergeableWeight': 787, 'montonicWeight': 234, 'totalValueWeight': 481},
        {'emptyWeight': 402, 'mergeableWeight': 582, 'montonicWeight': 268, 'totalValueWeight': 353},
        {'emptyWeight': 193, 'mergeableWeight': 327, 'montonicWeight': 397, 'totalValueWeight': 594},
        {'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 200},
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 200},
        {'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 200, 'totalValueWeight': 200},
        {'emptyWeight': 104, 'mergeableWeight': 922, 'montonicWeight': 787, 'totalValueWeight': 842},
        {'emptyWeight': 269, 'mergeableWeight': 212, 'montonicWeight': 236, 'totalValueWeight': 386},
        {'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 600},
        {'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 400, 'totalValueWeight': 600},
        {'emptyWeight': 757, 'mergeableWeight': 871, 'montonicWeight': 194, 'totalValueWeight': 558},
        {'emptyWeight': 52, 'mergeableWeight': 244, 'montonicWeight': 904, 'totalValueWeight': 951},
        {'emptyWeight': 517, 'mergeableWeight': 218, 'montonicWeight': 570, 'totalValueWeight': 990},
        {'emptyWeight': 732, 'mergeableWeight': 130, 'montonicWeight': 847, 'totalValueWeight': 387},
        {'emptyWeight': 608, 'mergeableWeight': 289, 'montonicWeight': 21, 'totalValueWeight': 155},
        {'emptyWeight': 19, 'mergeableWeight': 935, 'montonicWeight': 361, 'totalValueWeight': 894},
        {'emptyWeight': 398, 'mergeableWeight': 818, 'montonicWeight': 57, 'totalValueWeight': 39},
        {'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 200},
        {'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 600},
        {'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 800},
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 200, 'totalValueWeight': 400},
        {'emptyWeight': 902, 'mergeableWeight': 473, 'montonicWeight': 268, 'totalValueWeight': 770},
        {'emptyWeight': 879, 'mergeableWeight': 681, 'montonicWeight': 273, 'totalValueWeight': 535},
        {'emptyWeight': 997, 'mergeableWeight': 318, 'montonicWeight': 675, 'totalValueWeight': 545},
        {'emptyWeight': 561, 'mergeableWeight': 704, 'montonicWeight': 717, 'totalValueWeight': 733},
        {'emptyWeight': 333, 'mergeableWeight': 848, 'montonicWeight': 537, 'totalValueWeight': 775},
        {'emptyWeight': 584, 'mergeableWeight': 494, 'montonicWeight': 954, 'totalValueWeight': 952},
        {'emptyWeight': 992, 'mergeableWeight': 547, 'montonicWeight': 816, 'totalValueWeight': 746},
        {'emptyWeight': 475, 'mergeableWeight': 656, 'montonicWeight': 165, 'totalValueWeight': 775},
        {'emptyWeight': 716, 'mergeableWeight': 961, 'montonicWeight': 923, 'totalValueWeight': 856}
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
    print(f"Running scenarios {len(weights)} ({len(optimizationScenarios)} games)......")
    futures = [ _playOptimizationScenario.remote(op) for op in optimizationScenarios ]

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

    scenarios.sort(key=lambda op: op.valueTile(), reverse=True)
    print("===================================")
    print("===================================")
    print("Best Value Tiles:")
    for op in scenarios[:maxFitnessGroupSize]:
        print(op.report())
        print("===================================")

    scenarios.sort(key=lambda op: op.minTile(), reverse=True)
    print("===================================")
    print("===================================")
    print("Best Min Tiles:")
    for op in scenarios[:maxFitnessGroupSize]:
        print(op.report())
        print("===================================")

    scenarios.sort(key=lambda op: op.averageTile(), reverse=True)
    print("===================================")
    print("===================================")
    print("Best Averages:")
    for op in scenarios[:maxFitnessGroupSize]:
        print(op.report())
        print("===================================")

    scenarios.sort(key=lambda op: op.medianTile(), reverse=True)
    print("===================================")
    print("===================================")
    print("Best Medians:")
    for op in scenarios[:maxFitnessGroupSize]:
        print(op.report())
        print("===================================")

    with open('bestResultsStateSpace_GCP_SelectVersions.txt', 'w') as fout:
        for op in scenarios:
            print(op.report(), file=fout)
            print("----------------------------\n", file=fout)
            print("----------------------------\n", file=fout)
            
        scenarios.sort(key=lambda op: op.valueTile(), reverse=True)
        print("===================================", file=fout)
        print("===================================", file=fout)
        print("Best Value Tiles:", file=fout)
        for op in scenarios[:maxFitnessGroupSize]:
            print(op.report(), file=fout)
            print("===================================", file=fout)

        scenarios.sort(key=lambda op: op.minTile(), reverse=True)
        print("===================================", file=fout)
        print("===================================", file=fout)
        print("Best Min Tiles:", file=fout)
        for op in scenarios[:maxFitnessGroupSize]:
            print(op.report(), file=fout)
            print("===================================", file=fout)

        scenarios.sort(key=lambda op: op.averageTile(), reverse=True)
        print("===================================", file=fout)
        print("===================================", file=fout)
        print("Best Average Tiles:", file=fout)
        for op in scenarios[:maxFitnessGroupSize]:
            print(op.report(), file=fout)
            print("===================================", file=fout)

        scenarios.sort(key=lambda op: op.medianTile(), reverse=True)
        print("===================================", file=fout)
        print("===================================", file=fout)
        print("Best Median Tiles:", file=fout)
        for op in scenarios[:maxFitnessGroupSize]:
            print(op.report(), file=fout)
            print("===================================", file=fout)

    
    pEnd = time.time()
    print(f"Start: {pStart}")
    print(f"End: {pEnd}")


def _monotonicGeneticAlgorithm():
    pStart = time.time()
    algo = Algorithms['abexpecti']
    evaluator = Heuristics['monotonic']
    timePerMove = 0.15
    gamesPerIteration = 5
    crossoverCount = 50
    initialPopulationSize = 100  #start with a very large initial population
    maxFitnessGroupSize = 100
    maxIter = 25
    immigrantsPerGeneration = 4
    crossOverProbabilty = 0.9
    mutationProbability = 0.2
    minWeight = 0
    maxWeight = 1000
    displayGames = False
    population = [
        {'emptyWeight': 695, 'mergeableWeight': 812, 'montonicWeight': 727, 'totalValueWeight': 617},
        {'emptyWeight': 60, 'mergeableWeight': 679, 'montonicWeight': 942, 'totalValueWeight': 716},
        {'emptyWeight': 172, 'mergeableWeight': 294, 'montonicWeight': 245, 'totalValueWeight': 410},
        {'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 600},
        {'emptyWeight': 430, 'mergeableWeight': 829, 'montonicWeight': 895, 'totalValueWeight': 775},
        {'emptyWeight': 49, 'mergeableWeight': 566, 'montonicWeight': 723, 'totalValueWeight': 506},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 821, 'totalValueWeight': 911},
        {'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 400},
        {'emptyWeight': 210, 'mergeableWeight': 834, 'montonicWeight': 209, 'totalValueWeight': 763},
        {'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 865, 'mergeableWeight': 886, 'montonicWeight': 740, 'totalValueWeight': 515},
        {'emptyWeight': 713, 'mergeableWeight': 733, 'montonicWeight': 235, 'totalValueWeight': 999},
        {'emptyWeight': 922, 'mergeableWeight': 176, 'montonicWeight': 336, 'totalValueWeight': 194},
        {'emptyWeight': 107, 'mergeableWeight': 448, 'montonicWeight': 233, 'totalValueWeight': 365},
        {'emptyWeight': 406, 'mergeableWeight': 712, 'montonicWeight': 945, 'totalValueWeight': 628},
        {'emptyWeight': 134, 'mergeableWeight': 535, 'montonicWeight': 463, 'totalValueWeight': 776},
        {'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 828, 'mergeableWeight': 56, 'montonicWeight': 938, 'totalValueWeight': 660},
        {'emptyWeight': 124, 'mergeableWeight': 637, 'montonicWeight': 593, 'totalValueWeight': 236},
        {'emptyWeight': 387, 'mergeableWeight': 788, 'montonicWeight': 163, 'totalValueWeight': 528},
        {'emptyWeight': 445, 'mergeableWeight': 258, 'montonicWeight': 813, 'totalValueWeight': 736},
        {'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 400, 'totalValueWeight': 200},
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 400},
        {'emptyWeight': 691, 'mergeableWeight': 874, 'montonicWeight': 202, 'totalValueWeight': 241},
        {'emptyWeight': 631, 'mergeableWeight': 79, 'montonicWeight': 430, 'totalValueWeight': 776},
        {'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 600},
        {'emptyWeight': 718, 'mergeableWeight': 71, 'montonicWeight': 390, 'totalValueWeight': 669},
        {'emptyWeight': 434, 'mergeableWeight': 290, 'montonicWeight': 446, 'totalValueWeight': 678},
        {'emptyWeight': 533, 'mergeableWeight': 266, 'montonicWeight': 801, 'totalValueWeight': 824},
        {'emptyWeight': 723, 'mergeableWeight': 17, 'montonicWeight': 777, 'totalValueWeight': 321},
        {'emptyWeight': 601, 'mergeableWeight': 672, 'montonicWeight': 697, 'totalValueWeight': 571},
        {'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 800, 'totalValueWeight': 800},
        {'emptyWeight': 200, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 400},
        {'emptyWeight': 600, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 400},
        {'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 800, 'totalValueWeight': 800},
        {'emptyWeight': 565, 'mergeableWeight': 376, 'montonicWeight': 408, 'totalValueWeight': 679},
        {'emptyWeight': 800, 'mergeableWeight': 800, 'montonicWeight': 800, 'totalValueWeight': 400},
        {'emptyWeight': 368, 'mergeableWeight': 536, 'montonicWeight': 805, 'totalValueWeight': 204},
        {'emptyWeight': 655, 'mergeableWeight': 419, 'montonicWeight': 629, 'totalValueWeight': 911},
        {'emptyWeight': 5, 'mergeableWeight': 396, 'montonicWeight': 71, 'totalValueWeight': 75},
        {'emptyWeight': 367, 'mergeableWeight': 582, 'montonicWeight': 157, 'totalValueWeight': 347},
        {'emptyWeight': 39, 'mergeableWeight': 414, 'montonicWeight': 394, 'totalValueWeight': 147},
        {'emptyWeight': 468, 'mergeableWeight': 545, 'montonicWeight': 486, 'totalValueWeight': 264},
        {'emptyWeight': 405, 'mergeableWeight': 603, 'montonicWeight': 996, 'totalValueWeight': 652},
        {'emptyWeight': 362, 'mergeableWeight': 349, 'montonicWeight': 205, 'totalValueWeight': 799},
        {'emptyWeight': 481, 'mergeableWeight': 367, 'montonicWeight': 583, 'totalValueWeight': 567},
        {'emptyWeight': 504, 'mergeableWeight': 953, 'montonicWeight': 277, 'totalValueWeight': 766},
        {'emptyWeight': 375, 'mergeableWeight': 388, 'montonicWeight': 405, 'totalValueWeight': 731},
        {'emptyWeight': 23, 'mergeableWeight': 195, 'montonicWeight': 946, 'totalValueWeight': 531},
        {'emptyWeight': 908, 'mergeableWeight': 472, 'montonicWeight': 524, 'totalValueWeight': 574},
        {'emptyWeight': 604, 'mergeableWeight': 28, 'montonicWeight': 243, 'totalValueWeight': 546},
        {'emptyWeight': 776, 'mergeableWeight': 916, 'montonicWeight': 722, 'totalValueWeight': 574},
        {'emptyWeight': 669, 'mergeableWeight': 4, 'montonicWeight': 865, 'totalValueWeight': 465},
        {'emptyWeight': 742, 'mergeableWeight': 729, 'montonicWeight': 838, 'totalValueWeight': 589},
        {'emptyWeight': 227, 'mergeableWeight': 787, 'montonicWeight': 234, 'totalValueWeight': 481},
        {'emptyWeight': 402, 'mergeableWeight': 582, 'montonicWeight': 268, 'totalValueWeight': 353},
        {'emptyWeight': 193, 'mergeableWeight': 327, 'montonicWeight': 397, 'totalValueWeight': 594},
        {'emptyWeight': 400, 'mergeableWeight': 800, 'montonicWeight': 400, 'totalValueWeight': 200},
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 600, 'totalValueWeight': 800},
        {'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 200},
        {'emptyWeight': 200, 'mergeableWeight': 200, 'montonicWeight': 200, 'totalValueWeight': 200},
        {'emptyWeight': 104, 'mergeableWeight': 922, 'montonicWeight': 787, 'totalValueWeight': 842},
        {'emptyWeight': 269, 'mergeableWeight': 212, 'montonicWeight': 236, 'totalValueWeight': 386},
        {'emptyWeight': 800, 'mergeableWeight': 600, 'montonicWeight': 400, 'totalValueWeight': 600},
        {'emptyWeight': 400, 'mergeableWeight': 400, 'montonicWeight': 400, 'totalValueWeight': 600},
        {'emptyWeight': 757, 'mergeableWeight': 871, 'montonicWeight': 194, 'totalValueWeight': 558},
        {'emptyWeight': 52, 'mergeableWeight': 244, 'montonicWeight': 904, 'totalValueWeight': 951},
        {'emptyWeight': 517, 'mergeableWeight': 218, 'montonicWeight': 570, 'totalValueWeight': 990},
        {'emptyWeight': 732, 'mergeableWeight': 130, 'montonicWeight': 847, 'totalValueWeight': 387},
        {'emptyWeight': 608, 'mergeableWeight': 289, 'montonicWeight': 21, 'totalValueWeight': 155},
        {'emptyWeight': 19, 'mergeableWeight': 935, 'montonicWeight': 361, 'totalValueWeight': 894},
        {'emptyWeight': 398, 'mergeableWeight': 818, 'montonicWeight': 57, 'totalValueWeight': 39},
        {'emptyWeight': 200, 'mergeableWeight': 400, 'montonicWeight': 200, 'totalValueWeight': 200},
        {'emptyWeight': 800, 'mergeableWeight': 200, 'montonicWeight': 600, 'totalValueWeight': 600},
        {'emptyWeight': 400, 'mergeableWeight': 600, 'montonicWeight': 800, 'totalValueWeight': 800},
        {'emptyWeight': 600, 'mergeableWeight': 600, 'montonicWeight': 200, 'totalValueWeight': 400},
        {'emptyWeight': 902, 'mergeableWeight': 473, 'montonicWeight': 268, 'totalValueWeight': 770},
        {'emptyWeight': 879, 'mergeableWeight': 681, 'montonicWeight': 273, 'totalValueWeight': 535},
        {'emptyWeight': 997, 'mergeableWeight': 318, 'montonicWeight': 675, 'totalValueWeight': 545},
        {'emptyWeight': 561, 'mergeableWeight': 704, 'montonicWeight': 717, 'totalValueWeight': 733},
        {'emptyWeight': 333, 'mergeableWeight': 848, 'montonicWeight': 537, 'totalValueWeight': 775},
        {'emptyWeight': 584, 'mergeableWeight': 494, 'montonicWeight': 954, 'totalValueWeight': 952},
        {'emptyWeight': 992, 'mergeableWeight': 547, 'montonicWeight': 816, 'totalValueWeight': 746},
        {'emptyWeight': 475, 'mergeableWeight': 656, 'montonicWeight': 165, 'totalValueWeight': 775},
        {'emptyWeight': 716, 'mergeableWeight': 961, 'montonicWeight': 923, 'totalValueWeight': 856}
    ]
    
    with open('populationEvolution_GCP.txt', 'w') as fout:
        print("Population evolutions\n\n",file=fout)
    
    def toOptimizationScenario(weights:Dict[str, int], generation:int, games:int, display:bool):
        return OptimizationScenario(
            weights=weights,
            config=GameConfig(algo=algo, evaluator=evaluator(**weights), timePerTurn=timePerMove),
            generation=generation,
            games = games,
            display = display)

    for generation in range(maxIter):
        start = time.time()
        print(f"----------------------------")
        print(f"----------------------------")
        print(f"Generation {generation} (Population Size: {len(population)})...")

        population.extend(
            [ { 'emptyWeight': random.randint(minWeight, maxWeight), 
                'mergeableWeight': random.randint(minWeight, maxWeight), 
                'montonicWeight': random.randint(minWeight, maxWeight), 
                'totalValueWeight': random.randint(minWeight, maxWeight) }
                 for i in range(immigrantsPerGeneration) ]
        )

        """record new population"""
        with open('populationEvolution_GCP.txt', 'a') as fout:
            for p in population:
                print(p, file=fout)
                print(p)

        print("......")
        print("Running scenarios......")
        
        """run new population"""
        scenarios: List[OptimizationScenario] = [ toOptimizationScenario(w, 1, 1, displayGames) for w in population ]
        """run new population"""
        optimizationScenarios: List[OptimizationScenario] = [ toOptimizationScenario(w, 1, 1, displayGames) for w in population
                        for i in range(gamesPerIteration) ]
        print(f"Running scenarios {len(population)} ({len(optimizationScenarios)} games)......")
        futures = [ _playOptimizationScenario.remote(op) for op in optimizationScenarios ]

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

        """take top [maxFitnessGroupSize] fitnesses, write to file"""
        print("Recording results......")
        scenarios.sort(key=lambda op: op.valueTile(), reverse=True)
        scenarios = scenarios[:maxFitnessGroupSize]
        with open('populationResults_GCP.txt', 'w') as fout:
            print("Population results\n\n", file=fout)
            for op in scenarios:
                print(op.report(), file=fout)
                print("----------------------------\n", file=fout)
                print("----------------------------\n", file=fout)
        
        """get new population with individuals with top fitnesses"""
        print("Evolving population......")
        individuals = [op.weights for op in scenarios]
        reproductiveProbability = _getReproductiveProbabilityForIndividuals(scenarios)
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


def _playMultipleGamesAndReport(algorithm:str, heuristicName:str, games:int, timePerMove:float):
    algo = Algorithms[algorithm]
    evaluator = Heuristics[heuristicName]()
    config = GameConfig(algo=algo, evaluator=evaluator, timePerTurn=timePerMove)
    stats = []
    with multiprocessing.Pool() as pool:
        stats = pool.map(_playConfig, [config for i in range(games)])
    op = OptimizationScenario({}, config, 1, games, False)
    op.addStatistics(stats)
    print(f"--Config--\n{config}\n")
    print(f"--Statistics--\n{stats}\n")
    print(f"Scenario: ")
    print(op.report())
    
    
def _playMultipleGamesAndReportBase(games:int):
    stats = []
    print(f"Playing {games} games")
    with multiprocessing.Pool() as pool:
        stats = pool.map(_playGame, [False for i in range(games)])
    op = OptimizationScenario({}, getGameConfig(), 1, games, False)
    op.addStatistics(stats)
    print(f"--Config--\n{op.config}\n")
    print(f"--Statistics--\n{stats}\n")
    print(f"Scenario: ")
    print(op.report())
    

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
def geneticmonotonic():
    _monotonicGeneticAlgorithm()

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
@click.option('--games', '-g', default=1)
def playgame(algorithm, heuristic, timepermove, games):
    print(f'Algorithm: {algorithm}')
    print(f'Heuristics: {heuristic}')
    print(f'Time per move: {timepermove}')
    if games == 1:
        _playGameAndReport(algorithm, heuristic, timepermove)
    else:
        _playMultipleGamesAndReport(algorithm, heuristic, games, timepermove)


@run.command()
@click.option('--games', '-g', default=1)
def playdefaults(games):
    _playMultipleGamesAndReportBase(games)

if __name__ == "__main__":
    run()