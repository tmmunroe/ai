import click
import GameManager
import Grid
import IntelligentAgent
import multiprocessing as mp

from typing import List, Iterable, Sequence

from algos.games import CompositeEvaluator,Evaluator, GameAlgo, GameConfig, GameState, GameStatistics
from algos.evaluators import Monotonic, Snake, Corner
from algos.minimax import Minimax
from algos.alphabeta import AlphaBeta
from algos.expectiminimax import ExpectiMinimax
from algos.alphabetaExpecti import ExpectiAlphaBeta

Algorithms = {
    "random": GameAlgo,
    "minimax": Minimax,
    "alphabeta": AlphaBeta,
    "expecti": ExpectiMinimax,
    "abexpecti": ExpectiAlphaBeta
}

Heuristics = {
    "monotonic": Monotonic,
    "snake": Snake,
    "corner": Corner
}

DefaultTimePerMove = 0.18
DefaultAlgo = 'abexpecti'
DefaultHeuristics = ['monotonic']


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


def getTestGrids() -> Sequence[Grid.Grid]:
    grids: List[Grid.Grid] = []
    grids.append(
        gridFor([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ])
    )

    grids.extend(getCornerGrids())
    grids.extend(getMiddleGrids())
    grids.extend(getMonotonicGrids())
    grids.extend(getMergingGrids())


    return grids


def testHeuristic(heuristicNames: Sequence[str]):
    heuristicName = heuristicNames[0]
    heuristic = Heuristics[heuristicName]()
    testGrids = getTestGrids()
    for grid in testGrids:
        print("===================================")
        print("===================================")
        printBoardAndValue(grid, heuristicName, heuristic)


def _playGame(algorithm:str, heuristicNames:Iterable[str], weights:Iterable[float], timePerMove:float) -> GameStatistics:
    algo = Algorithms[algorithm]
    heuristics = [ Heuristics[h]() for h in heuristicNames ]
    evaluator = None

    if len(heuristics) == 1:
        evaluator = heuristics[0]
    else:
        evaluator = CompositeEvaluator(heuristics, weights)

    config = GameConfig(algo=algo, evaluator=evaluator, timePerTurn=timePerMove)
    
    IntelligentAgent.setGameConfig(config)
    GameManager.main()
    return IntelligentAgent.Statistics


def testWeights(algorithm:str, heuristicNames:Iterable[str], weights:Iterable[float],timePerMove:float, games:int):
    allStats = []
    playHeuristics = []
    playWeights = []
    for h,w in zip(heuristicNames, weights):
        if w != 0:
            playHeuristics.append(h)
            playWeights.append(w)

    for game in range(games):
        s = _playGame(algorithm, playHeuristics, playWeights, timePerMove)
        s.processFinalState()
        allStats.append(s)

    averageMax = sum((s.maxTile for s in allStats)) / len(allStats)
    return averageMax, weights


def _optimize(algorithm:str, timePerMove:float, games:int):
    weightRange = list(range(2))
    heuristicNames = list(Heuristics.keys())
    allCombinations = [ (a,b,c,d) 
            for a in weightRange 
            for b in weightRange
            for c in weightRange
            for d in weightRange ]
    results = []
    for weights in allCombinations:
        if all((w == 0 for w in weights)):
            continue
        results.append(testWeights(algorithm, heuristicNames, weights, timePerMove, games))

    results.sort()
    print(results[:50])

def _playGameAndReport(algorithm:str, heuristicNames:Iterable[str], weights:Iterable[float], timePerMove:float):
    statistics = _playGame(algorithm, heuristicNames, weights, timePerMove)
    print(IntelligentAgent.Statistics)


@click.group()
def run():
    pass


@run.command()
@click.option('--heuristic', '-h',
    default=DefaultHeuristics,
    type=click.Choice(list(Heuristics), case_sensitive=True),
    multiple=True)
def checkheuristic(heuristic):
    testHeuristic(heuristic)

@run.command()
@click.option('--algorithm', '-a', 
    default=DefaultAlgo,
    type=click.Choice(list(Algorithms), case_sensitive=True))
@click.option('--heuristic', '-h',
    default=DefaultHeuristics,
    type=click.Choice(list(Heuristics), case_sensitive=True),
    multiple=True)
@click.option('--weight', '-w',
    default=[1],
    type=click.FLOAT,
    multiple=True)
@click.option('--timepermove', '-t', default=DefaultTimePerMove)
def playgame(algorithm, heuristic, weight, timepermove):
    print(f'Algorithm: {algorithm}')
    print(f'Heuristics: {heuristic}')
    print(f'Weights: {weight}')
    print(f'Time per move: {timepermove}')
    if len(heuristic) != len(weight):
        raise Exception("Heuristic count should be the same as weight count")
    _playGameAndReport(algorithm, heuristic, weight, timepermove)


@run.command()
@click.option('--algorithm', '-a', 
    default=DefaultAlgo,
    type=click.Choice(list(Algorithms), case_sensitive=True))
@click.option('--timepermove', '-t', default=DefaultTimePerMove)
@click.option('--games', '-g', default=1)
def optimize(algorithm, timepermove, games):
    _optimize(algorithm, timepermove, games)


if __name__ == "__main__":
    run()