
from collections import deque
import heapq
from functools import total_ordering

def recover_path(target, start, previous_nodes):
    path = [target]
    node = target
    while node != start:
        node = previous_nodes[node]
        path.append(node)
    return list(reversed(path))
        

def dfs(adjacency_list: dict, start:str, target: str):
    '''returns the path to target, and exploration order'''
    deq: deque = deque()
    previous_nodes: dict = dict()
    explored: list = list()

    deq.append(start)
    previous_nodes[start] = None

    while len(deq) != 0:
        exploring = deq.pop()
        explored.append(exploring)

        if exploring == target:
            path = recover_path(target, start, previous_nodes)
            return path, explored

        children = adjacency_list[exploring]
        for child in sorted(children, reverse=True):
            if child not in explored and child not in deq:
                deq.append(child)
                previous_nodes[child] = exploring

    return [], explored


def bfs(adjacency_list: dict, start:str, target: str):
    '''returns the path to target, and exploration order'''
    deq: deque = deque()
    previous_nodes: dict = dict()
    explored: list = list()

    deq.appendleft(start)
    previous_nodes[start] = None

    while len(deq) != 0:
        exploring = deq.pop()
        explored.append(exploring)

        if exploring == target:
            path = recover_path(target, start, previous_nodes)
            return path, explored

        children = adjacency_list[exploring]
        for child in sorted(children):
            if child not in explored and child not in deq:
                deq.appendleft(child)
                previous_nodes[child] = exploring

    return [], explored

@total_ordering
class NodeCost:
    def __init__(self, id, cost):
        self.id = id
        self.cost = cost

    def __eq__(self, other):
        return (self.cost == other.cost) and (self.id == other.id)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if self.cost == other.cost:
            return self.id < other.id
        return self.cost < other.cost

def ucs(adjacency_list_with_costs: dict, start:str, target: str):
    '''returns the path to target, and exploration order'''
    q: list = list()
    previous_nodes: dict = dict()
    seen_nodes: dict = dict()
    explored: list = list()

    previous_nodes[start] = None
    nc = NodeCost(start, 0)
    heapq.heappush(q, nc)
    seen_nodes[start] = nc

    while len(q) != 0:
        exploring = heapq.heappop(q)
        explored.append(exploring.id)

        if exploring.id == target:
            path = recover_path(target, start, previous_nodes)
            return path, explored

        children = adjacency_list_with_costs[exploring.id]
        for child, cost in children.items():
            if child in explored:
                continue

            cost_from_me = exploring.cost + cost
            if child in seen_nodes:
                nc = seen_nodes[child]
                if cost_from_me < nc.cost:
                    previous_nodes[child] = exploring.id
                    nc.cost = cost_from_me
            else:
                nc = NodeCost(child, cost_from_me)
                seen_nodes[child] = nc
                previous_nodes[child] = exploring.id
                q.append(nc)

        heapq.heapify(q)

    return [], explored


def astar(adjacency_list_with_costs: dict, heuristics: dict, start:str, target: str):
    '''returns the path to target, and exploration order'''
    q: list = list()
    previous_nodes: dict = dict()
    seen_nodes: dict = dict()
    explored: list = list()

    previous_nodes[start] = None
    nc = NodeCost(start, heuristics[start])
    heapq.heappush(q, nc)
    seen_nodes[start] = nc

    while len(q) != 0:
        exploring = heapq.heappop(q)
        explored.append(exploring.id)

        print()
        print(f"Exploring {exploring.id}")
        if exploring.id == target:
            path = recover_path(target, start, previous_nodes)
            return path, explored

        children = adjacency_list_with_costs[exploring.id]
        for child, cost in children.items():
            if child in explored:
                continue

            explorerTravelCost = exploring.cost - heuristics[exploring.id]
            gn = explorerTravelCost + cost
            hn = heuristics[child]
            cost_from_me = gn + hn

            print(f"{exploring.id} -> {child}: f({child}) = g({child}) + h({child}) = {gn} + {hn} = {gn + hn}")

            if child in seen_nodes:
                nc = seen_nodes[child]
                if cost_from_me < nc.cost:
                    previous_nodes[child] = exploring.id
                    nc.cost = cost_from_me
            else:
                nc = NodeCost(child, cost_from_me)
                seen_nodes[child] = nc
                previous_nodes[child] = exploring.id
                q.append(nc)

        heapq.heapify(q)

    return [], explored

def main():
    adj_list = {
        "A": ["B", "C"],
        "B": [],
        "C": ["D", "E", "F"],
        "D": ["G", "H"],
        "E": [],
        "F": [],
        "G": ["I"],
        "H": [],
        "I": []
    }

    print("DFS:")
    path, explored_order = dfs(adj_list, "A", "F")
    print("Path:")
    print(path)
    print()
    print("Explored Order:")
    print(explored_order)


    print()
    print("BFS:")
    path, explored_order = bfs(adj_list, "A", "F")
    print("Path:")
    print(path)
    print()
    print("Explored Order:")
    print(explored_order)


    
    adj_costs = {
        "B": {"E": 4},
        "C": {"D": 2},
        "D": {"G": 4, "S": 6},
        "E": {"F": 4, "Z": 8, "G": 2},
        "F": {"B": 3},
        "G": {"Z": 5},
        "S": {"B": 2, "C": 3, "F": 4},
        "Z": {}
    }

    print()
    print()
    print("UCS:")
    path, explored_order = ucs(adj_costs, "S", "Z")
    print("Path:")
    print(path)
    print()
    print("Explored Order:")
    print(explored_order)

    
    heuristics = {
        "S": 8,
        "B": 7,
        "C": 6,
        "D": 5,
        "E": 4,
        "F": 5,
        "G": 2,
        "Z": 0
    }

    print()
    print()
    print("A*:")
    path, explored_order = astar(adj_costs, heuristics, "S", "Z")
    print("Path:")
    print(path)
    print()
    print("Explored Order:")
    print(explored_order)


if __name__ == '__main__':
    main()