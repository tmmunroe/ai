import click
import GameManager
import IntelligentAgent

from typing import Iterable

from algos.games import CompositeEvaluator, GameAlgo, GameConfig
from algos.evaluators import EmptySpaces, MaxTile
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
    "maxtile": MaxTile,
    "emptyspaces": EmptySpaces
}


def _playGame(algorithm:str, heuristicNames:Iterable[str], weights:Iterable[float], timePerMove:float):
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

    print("GameStatistics:")
    print(IntelligentAgent.Statistics)


@click.command()
@click.option('--algorithm', '-a', 
    default='minimax',
    type=click.Choice(list(Algorithms), case_sensitive=True))
@click.option('--heuristic', '-h',
    default=['maxtile', 'emptyspaces'],
    type=click.Choice(list(Heuristics), case_sensitive=True),
    multiple=True)
@click.option('--weight', '-w',
    default=[0.5, 0.5],
    type=click.FLOAT,
    multiple=True)
@click.option('--timepermove', '-t',
    default=0.2)
def playgame(algorithm, heuristic, weight, timepermove):
    print(f'Algorithm: {algorithm}')
    print(f'Heuristics: {heuristic}')
    print(f'Weights: {weight}')
    print(f'Time per move: {timepermove}')
    if len(heuristic) != len(weight):
        raise Exception("Heuristic count should be the same as weight count")
    _playGame(algorithm, heuristic, weight, timepermove)


if __name__ == "__main__":
    playgame()