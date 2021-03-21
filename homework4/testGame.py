import click
import GameManager
import Grid
import IntelligentAgent
import ComputerAI
import Displayer
import BaseDisplayer
import random
import concurrent.futures

from typing import Dict, List, Iterable, Sequence, Tuple

from algos.games import CompositeEvaluator,Evaluator, GameAlgo, GameConfig, GameState, GameStatistics
from algos.evaluators import Monotonic, Snake, Corner
from algos.minimax import Minimax
from algos.alphabeta import AlphaBeta
from algos.expectiminimax import ExpectiMinimax
from algos.alphabetaExpecti import ExpectiAlphaBeta


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


def averageTile(stats: Sequence[GameStatistics]) -> float:
    return sum([stat.maxTile for stat in stats]) / len(stats)

class OptimizationScenario:
    def __init__(self, config:GameConfig, games:int, display:bool):
        self.config = config
        self.games = games
        self.display = display
    
    def __str__(self):
        return f"{self.config} ({self.games} games)"


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


def testHeuristic(heuristicNames: Sequence[str]):
    heuristicName = heuristicNames[0]
    heuristic = Heuristics[heuristicName]()
    testGrids = getTestGrids(heuristicName)
    for testSet, grids in testGrids.items():
        print(f"================={testSet}===============")
        for grid in grids:
            printBoardAndValue(grid, heuristicName, heuristic)
            print("===================================")

def _playConfig(config: GameConfig, display:bool = True) -> GameStatistics:
    #GameManager.main()
    intelligentAgent = IntelligentAgent.IntelligentAgent(config=config)
    computerAI  = ComputerAI.ComputerAI()
    displayer   = Displayer.Displayer() if display else BaseDisplayer.BaseDisplayer()
    gameManager = GameManager.GameManager(4, intelligentAgent, computerAI, displayer)

    maxTile     = gameManager.start()
    print(maxTile)

    return intelligentAgent.Statistics


def _playMultipleGames(config: GameConfig, games:int = 3, display:bool = True) -> Sequence[GameStatistics]:
    stats = []
    for game in range(games):
        stats.append(_playConfig(config, display=display))
    return stats

def _playOptimizationScenario(op: OptimizationScenario) -> Tuple[OptimizationScenario, Sequence[GameStatistics]]:
    return op, _playMultipleGames(op.config, op.games, op.display)

def _optimizeMonotonic():
    algo = Algorithms['abexpecti']
    evaluator = Heuristics['monotonic']
    timePerMove = 0.01
    gamesPerIteration = 3
    populationSize = 8
    crossoverProbability = 0.8
    mutationProbability = 0.1
    minWeight = 0
    maxWeight = 1000
    displayGames = False
    population = [
        {'emptyWeight':random.randint(minWeight, maxWeight),
         'mergeableWeight':random.randint(minWeight, maxWeight),
         'montonicWeight':random.randint(minWeight, maxWeight),
         'totalValueWeight':random.randint(minWeight, maxWeight) }
         for i in range(populationSize)
    ]

    with open('initialPopulation.txt', 'w') as fout:
        for p in population:
            print(p, file=fout)
    
    def toOptimizationScenario(weights:Dict[str, int], games:int, display:bool):
        return OptimizationScenario(
            config=GameConfig(algo=algo, evaluator=evaluator(**weights), timePerTurn=timePerMove),
            games = games,
            display = display)

    best = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        optimizationScenarios = [ toOptimizationScenario(w, gamesPerIteration, displayGames) for w in population ]
        futures = [ executor.submit(_playOptimizationScenario, op) for op in optimizationScenarios ]
        
        for future in concurrent.futures.as_completed(futures):
            scenario, stats = future.result()
            for stat in stats:
                stat.processFinalState()
            best.append((scenario, stats))
        
        best.sort(key=lambda scenStats: averageTile(scenStats[1]), reverse=True)
        with open('results.txt', 'w') as fout:
            for scenario, stats in best:
                print(f"{scenario}\nAverageMaxTile: {averageTile(stats)}")
                print(f"{scenario}\nAverageMaxTile: {averageTile(stats)}", file=fout)
                for i, stat in enumerate(stats):
                    print(f"Statistics: Game {i}", file=fout)
                    print(stat, file=fout)
                    print(file=fout)


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
def optimizemonotonic():
    _optimizeMonotonic()


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