
from __future__ import division
from __future__ import print_function

import sys
import math
import time
import resource
from collections import deque
import heapq
from functools import total_ordering

GoalState = None
GoalStateHash = None
GoalStateRange = None

def hashConfig(config: list):
    return hash(tuple(config))

class SearchStatistics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.path_to_goal = None
        self.nodes_expanded = 0
        self.max_search_depth = 0
        self.resource_usage = None
        self.truncatePath = False

    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.end_time = time.time()

    @property
    def cost_of_path(self):
        if self.path_to_goal:
            return len(self.path_to_goal)
        return 0
    
    @property
    def search_depth(self):
        return self.cost_of_path

    @property
    def running_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    @property
    def max_ram_usage(self):
        if self.resource_usage:
            return self.resource_usage.ru_maxrss / 1000.0
        return 0

    def reset(self):
        self.path_to_goal = None
        self.nodes_expanded = 0
        self.max_search_depth = 0
        self.resource_usage = None

    def report(self):
        if len(self.path_to_goal) > 20 and self.truncatePath:
            path = self.path_to_goal[:5]
            path.append('...')
            path.extend(self.path_to_goal[-5:])
        else:
            path = self.path_to_goal
    
        reportOutput = [
        f'path_to_goal: {path}',
        f'cost_of_path: {self.cost_of_path}',
        f'nodes_expanded: {self.nodes_expanded}',
        f'search_depth: {self.search_depth}',
        f'max_search_depth: {self.max_search_depth}',
        f'running_time: {self.running_time:.8f}',
        f'max_ram_usage: {self.max_ram_usage:.8f}'
        ]
        return '\n'.join(reportOutput)

#### SKELETON CODE ####
## The Class that Represents the Puzzle
@total_ordering
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n         = n
        self.max_index = n*n - 1
        self.cost      = cost
        self.parent    = parent
        self.action    = action
        self.config    = config
        self.children  = []
        self.hash      = hashConfig(self.config)

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def my_size(self):
        print()
        print(f"self: {sys.getsizeof(self)}")
        print(f"n: {sys.getsizeof(self.n)}")
        print(f"max_index: {sys.getsizeof(self.max_index)}")
        print(f"cost: {sys.getsizeof(self.cost)}")
        print(f"parent: {sys.getsizeof(self.parent)}")
        print(f"action: {sys.getsizeof(self.action)}")
        print(f"config: {sys.getsizeof(self.config)}")
        print(f"config contents: {sum([sys.getsizeof(i) for i in self.config])}")
        print(f"children: {sys.getsizeof(self.children)}")
        print(f"children contents: {sum([sys.getsizeof(i) for i in self.children])}")
        print(f"hash: {sys.getsizeof(self.hash)}")
        print(f"blank_index: {sys.getsizeof(self.blank_index)}")
        total = (
                sys.getsizeof(self) +
                sys.getsizeof(self.n) +
                sys.getsizeof(self.max_index) +
                sys.getsizeof(self.cost) +
                sys.getsizeof(self.parent) +
                sys.getsizeof(self.action) +
                sys.getsizeof(self.config) +
                sys.getsizeof(self.children) +
                sys.getsizeof(self.hash) +
                sys.getsizeof(self.blank_index))
        print(f"total: {total}")
        print()
        
        return total

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if hash(self) == hash(other):
            return self.config == other.config
        return False

    def __lt__(self, other):
        if hash(self) < hash(other):
            return True
        elif hash(self) > hash(other):
            return False
        else:
            return self.config < other.config

    def printReport(self):
        print(f'action: {self.action} config: {self.config} hash: {hash(self)} cost: {self.cost}')
        self.display()

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[self.n*i : self.n*(i+1)])

    def _is_valid_index(self, i):
        return 0 <= i <= self.max_index

    def _new_puzzle_state(self, actionShift, actionName):
        new_blank_index = self.blank_index + actionShift
        if not self._is_valid_index(new_blank_index):
            return None
        new_config = self.config[:]
        new_config[new_blank_index], new_config[self.blank_index] = new_config[self.blank_index], new_config[new_blank_index]
        return PuzzleState(new_config, self.n, self, actionName, self.cost+1)

    def _column(self, index):
        #0-indexed column
        if index == 0:
            return 0
        return index % self.n

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        return self._new_puzzle_state(-self.n, 'U')
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        return self._new_puzzle_state(self.n, 'D')
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        if self._column(self.blank_index) == 0:
            return None
        return self._new_puzzle_state(-1, 'L')

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        if self._column(self.blank_index) == (self.n - 1):
            return None
        return self._new_puzzle_state(1, 'R')
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children


def pretty_action_name(action):
    first = action[0]
    if first == 'D':
        return 'Down'
    elif first == 'U':
        return 'Up'
    elif first == 'L':
        return 'Left'
    elif first == 'R':
        return 'Right'

def recover_path(state:PuzzleState):
    path = []
    s = state
    while True:
        if s.action == 'Initial':
            break
        path.append(pretty_action_name(s.action))
        s = s.parent
    path.reverse()
    return path

def updateStatsForExpanded(state:PuzzleState, searchStats: SearchStatistics):
    searchStats.nodes_expanded += 1

def updateStatsForSeen(state:PuzzleState, searchStats: SearchStatistics):
    searchStats.max_search_depth = max(state.cost, searchStats.max_search_depth)

def updateStatsForGoalState(state:PuzzleState, searchStats: SearchStatistics):
    searchStats.path_to_goal = recover_path(state)
    searchStats.resource_usage = resource.getrusage(resource.RUSAGE_SELF)

def initializeGoalState(initialState: PuzzleState):
    global GoalState
    global GoalStateHash
    global GoalStateRange
    GoalState = list(range(len(initialState.config)))
    GoalStateHash = hashConfig(GoalState)
    GoalStateRange = set(GoalState)
    '''
    print()
    print(f'InitialState: {initialState.config}')
    print(f'GoalState: {GoalState}')
    print(f'GoalStateHash: {GoalStateHash}')
    '''


### Students need to change the method to have the corresponding parameters
def writeOutput(searchStats: SearchStatistics):
    with open('output.txt', 'w') as f:
        print(searchStats.report(), file=f)


def bfs_search(initial_state):
    """BFS search"""
    initializeGoalState(initial_state)
    searchStats = SearchStatistics()
    searchStats.start()
    seen = set()
    frontier_queue = deque()

    def popState():
        s = frontier_queue.popleft()
        return s
    
    def pushState(s):
        frontier_queue.append(s)
        seen.add(s)

    pushState(initial_state)

    while len(frontier_queue) > 0:
        '''pop node'''
        node = popState()

        '''check if node satisfies goal'''
        if test_goal(node):
            updateStatsForGoalState(node, searchStats)
            searchStats.end()
            writeOutput(searchStats)
            break
        
        '''expand node'''
        updateStatsForExpanded(node, searchStats)
        children = node.expand()

        '''for each child'''
        for child in children:
            '''if child not in seen, push'''
            if child in seen:
                continue

            updateStatsForSeen(child, searchStats)
            pushState(child)

def dfs_search(initial_state):
    """DFS search"""
    initializeGoalState(initial_state)
    searchStats = SearchStatistics()
    searchStats.start()
    seen = set()
    frontier_queue = deque()

    def popState():
        s = frontier_queue.pop()
        return s
    
    def pushState(s):
        frontier_queue.append(s)
        seen.add(s)

    pushState(initial_state)

    while len(frontier_queue) > 0:
        '''pop node'''
        node = popState()

        '''check if node satisfies goal'''
        if test_goal(node):
            updateStatsForGoalState(node, searchStats)
            searchStats.end()
            writeOutput(searchStats)
            break
        
        '''expand node'''
        updateStatsForExpanded(node, searchStats)
        children = node.expand()

        '''for each child'''
        for child in reversed(children):
            '''if child not in seen, push'''
            if child in seen:
                continue

            updateStatsForSeen(child, searchStats)
            pushState(child)

def A_star_search(initial_state):
    """A * search"""
    initializeGoalState(initial_state)
    searchStats = SearchStatistics()
    searchStats.start()
    explored = set()
    frontier = dict()
    frontier_queue = []
    epoch = 0

    def popState():
        _, _, s = heapq.heappop(frontier_queue)
        del frontier[s]
        explored.add(s)
        return s
    
    def pushState(s):
        nonlocal epoch
        epoch = epoch + 1
        cost = calculate_total_cost(s)
        queue_item = [ cost, epoch, s ]
        heapq.heappush(frontier_queue,queue_item)
        frontier[s] = queue_item

    def updatePriority(s, newCost):
        if s.cost <= newCost:
            return
        s.cost = newCost
        newTotalCost = calculate_total_cost(s)
        queue_item = frontier[s]
        queue_item[0] = newTotalCost
        queue_is_sorted = False
    
    def reheap():
        heapq.heapify(frontier_queue)
        queue_is_sorted = True

    pushState(initial_state)
    queue_is_sorted = True

    while len(frontier_queue) > 0:
        '''maintain heap'''
        if not queue_is_sorted:
            reheap()

        '''pop node'''
        node = popState()

        '''check if node satisfies goal'''
        if test_goal(node):
            updateStatsForGoalState(node, searchStats)
            searchStats.end()
            writeOutput(searchStats)
            break
        
        '''expand node'''
        updateStatsForExpanded(node, searchStats)
        children = node.expand()

        '''for each child'''
        for child in children:
            '''if child not in seen, push'''
            if child in explored:
                continue

            if child in frontier:
                updatePriority(child, node.cost + 1)
                continue

            updateStatsForSeen(child, searchStats)
            pushState(child)


def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile
    idx: current location on the 1-d board
    value: current value at idx
    n: the board dimension"""
    if value == 0:
        return 0
    goal_col, goal_row = divmod(value, n)
    curr_col, curr_row = divmod(idx, n)
    return abs(goal_col - curr_col) + abs(goal_row - curr_row)

def calculate_manhattan_cost(state: PuzzleState):
    total = 0
    for idx, value in enumerate(state.config):
        total += calculate_manhattan_dist(idx, value, state.n)
    return total

def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    return state.cost + calculate_manhattan_cost(state)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    if hash(puzzle_state) == GoalStateHash:
        return puzzle_state.config == GoalState
    return False


def findPath(search_mode:str, begin_state:list):
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))

    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()

    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")

    end_time = time.time()
    print("Search completed in %.3f second(s)"%(end_time-start_time))

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    '''
    r = resource.getrusage(resource.RUSAGE_SELF)
    print(f'Beginning size: {r.ru_maxrss / 1000.0}')
    '''

    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")

    findPath(search_mode, begin_state)

if __name__ == '__main__':
    main()
