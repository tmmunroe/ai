
from algos.games import GameConfig
from algos.evaluators import Monotonic, Snake, Corner
from algos.alphabetaExpecti import ExpectiAlphaBeta

DefaultAlgo = ExpectiAlphaBeta
DefaultEvaluator = Monotonic
DefaultTimePerMove = 0.18

def getGameConfig():
    return GameConfig(algo=DefaultAlgo, evaluator=DefaultEvaluator(), timePerTurn=DefaultTimePerMove)