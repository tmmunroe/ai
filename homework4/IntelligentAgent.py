import concurrent.futures
from BaseAI import BaseAI
from algos.games import GameConfig, GameState, GameStatistics, NullAction

Config: GameConfig = GameConfig()
Statistics: GameStatistics = GameStatistics()
Executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def setGameConfig(config:GameConfig):
	global Config
	Config = config

class IntelligentAgent(BaseAI):
	def getMove(self, grid):
		gridState = GameState(grid)
		Statistics.updateState(gridState)
		algo = Config.Algo(gridState, Config.Evaluator, Statistics)
		s = Executor.submit(algo.search)
		a = None
		try:
			a = s.result(timeout=Config.TimePerTurn)
		except concurrent.futures.TimeoutError:
			a = algo.terminateSearch()
		return a if a != NullAction else None