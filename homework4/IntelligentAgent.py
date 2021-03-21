import concurrent.futures
from BaseAI import BaseAI
from algos.games import GameConfig, GameState, GameStatistics, NullAction
from algos.config import getGameConfig

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
		a = None
		try:
			a = s.result(timeout=self.Config.TimePerTurn)
		except concurrent.futures.TimeoutError:
			a = algo.terminateSearch()

		self.Statistics.updateCacheStats(self.HeuristicCache)
		return a if a != NullAction else None