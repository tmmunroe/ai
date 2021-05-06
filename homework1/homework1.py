"""
Name: <your_name>
Uni: <your_uni>
"""

# The function signatures are named for the questions they represent, for
# instance p1 is problem 1. Please do not modify any of the function signatures
# as they will be used for grading purposes

import numpy as np

def p1(k: int) -> str:
    s, factorial = '1', 1
    for i in range(2, k+1):
        factorial *= i
        s = f'{factorial},{s}'
    return s


def p2_a(x: list, y: list) -> list:
    """Return the resulting list after the following operations:  
    (i) sort y by descending order,  and 
    (ii) delete the last element of this sorted list"""
    new_y = sorted(y, reverse=True)
    return new_y[:-1]


def p2_b(x: list, y: list) -> list:
    """Return the reverse list of x"""
    return x[::-1]


def p2_c(x: list, y: list) -> list:
    """Return the list of unique elements that you get by concatenating x and y, in ascending order"""
    unique_xy = set()
    unique_xy.update(x)
    unique_xy.update(y)
    xy = list(unique_xy)
    xy.sort()
    return xy


def p2_d(x: list, y: list) -> list:
    """Return a single list consisting of x and y as its two elements"""
    return [x, y]


def p3_a(x: set, y: set, z: set) -> set:
    """Return the union set of all three sets"""
    return x | y | z


def p3_b(x: set, y: set, z: set) -> set:
    """Return the intersection set of all three sets"""
    return x & y & z


def p3_c(x: set, y: set, z: set) -> set:
    """Return a set of elements that belong to only a single set out of the three sets"""
    all_elements = p3_a(x,y,z)
    multi_elements = p3_a(x&y, x&z, y&z)
    return all_elements - multi_elements

def p4_a() -> np.array:
    """return a 5x5 array with the following values:
      i) edges and corners populated with 1
      ii) center populated with 2
      iii) everything else populated with 0"""
    return np.array([
        [ 1, 1, 1, 1, 1 ],
        [ 1, 0, 0, 0, 1 ],
        [ 1, 0, 2, 0, 1 ],
        [ 1, 0, 0, 0, 1 ],
        [ 1, 1, 1, 1, 1 ]
    ])


def p4_b(x: np.array) -> list:
    PAWN, KNIGHT, EMPTY = 2, 1, 0
    ROW, COL = 0,1
    MIN_POS, MAX_POS = (0,0), (4,4)
    MOVES = [
        (  1,  2 ),
        (  1, -2 ),
        ( -1,  2 ),
        ( -1, -2 ),
        (  2,  1 ),
        (  2, -1 ),
        ( -2,  1 ),
        ( -2, -1 )
    ]

    def valid_pos(position: tuple, min_pos: tuple, max_pos: tuple) -> bool:
        return ((min_pos[ROW] <= position[ROW] <= max_pos[ROW]) and
                (min_pos[COL] <= position[COL] <= max_pos[COL]))

    def new_pos(position: tuple, move: tuple) -> tuple:
        return (position[ROW] + move[ROW], position[COL] + move[COL])

    def knight_moves(position: tuple, min_pos: tuple, max_pos: tuple) -> list:
        end_positions = (new_pos(position, move) for move in MOVES)
        return [pos for pos in end_positions if valid_pos(pos, min_pos, max_pos)]
    
    def find_pieces(x: np.array, target: int) -> list:
        positions = []
        rows = len(x)
        columns = len(x[0])
        for row in range(rows):
            for col in range(columns):
                if x[row][col] == target:
                    positions.append((row, col))
        return positions

    def knight_positions(x: np.array) -> list:
        return find_pieces(x, KNIGHT)

    def pawn_position(x: np.array) -> tuple:
        piece_positions = find_pieces(x, PAWN)
        if len(piece_positions) != 1:
            raise Exception(f"Unexpected number of pawns {len(piece_positions)}")
        return piece_positions[0]

    pawn = pawn_position(x)

    knights = knight_positions(x)
    knight_moves_from_pawn = knight_moves(pawn, MIN_POS, MAX_POS)

    knights_attacking_pawn = set(knights).intersection(knight_moves_from_pawn)

    return list(knights_attacking_pawn)


def p5_a(x: dict) -> int:
    "Return the number of isolated nodes (i.e.  nodes without any connections)"
    return len(x) - p5_b(x)


def p5_b(x: dict) -> int:
    "Return the number of non-isolated nodes (i.e.  nodes with connections)"
    nodes_with_neighbors = [ node for node,neighbors in x.items() if neighbors and len(neighbors) > 0 ]
    return len(nodes_with_neighbors)

def p5_c(x: dict) -> list:
    def directed_edge(a, b):
        if a < b:
            return a,b
        return b,a
    
    unique_edges = set()
    for node, neighbors in x.items():
        my_edges = [ directed_edge(node, neighbor) for neighbor in neighbors ]
        unique_edges.update(my_edges)
    
    return list(unique_edges)


def p5_d(x: dict) -> np.array:
    dim = len(x)
    adjacency_matrix = np.array([ [ 0 for i in range(dim) ] for j in range(dim) ])
    #because according to piazza, the adjacency matrix indices should correspond to sorted order of the nodes(keys)
    sorted_keys = sorted(x.keys()) 
    node_indices = { node: index for index,node in enumerate(sorted_keys) }
    
    edges = p5_c(x)
    for node_a,node_b in edges:
        index_a = node_indices[node_a]
        index_b = node_indices[node_b]
        adjacency_matrix[index_a][index_b] = adjacency_matrix[index_b][index_a] = 1

    return adjacency_matrix


class Node:
    def __init__(self, key, value, prevNode=None, nextNode=None):
        self.key = key
        self.value = value
        self.prevNode = prevNode
        self.nextNode = nextNode

#Question 6
class PriorityQueue(object):
    priorities = {
        "apple"     : 5.0, 
        "banana"    : 4.5,
        "carrot"    : 3.3,
        "kiwi"      : 7.4,
        "orange"    : 5.0,
        "mango"     : 9.1,
        "pineapple" : 9.1
    }

    def __init__(self):
        '''create a new priority queue'''
        self.front, self.back = None, None
    
    def _priority(self, key):
        return self.priorities[key]

    def push(self, x):
        '''push x onto the queue'''
        p_x = self._priority(x)

        if not self.front:
            self.front = self.back = Node(x, p_x)
            return
        
        node = self.front
        while node:
            if p_x <= node.value:
                newNode = Node(x, p_x, prevNode=node.prevNode, nextNode=node)
                node.prevNode = newNode
                if node == self.front:
                    self.front = newNode
                return
            node = node.nextNode

        self.back.nextNode = Node(x, p_x, prevNode=self.back, nextNode=None)
        self.back = self.back.nextNode

    def pop(self):
        '''pop and return the highest priority item from the queue'''
        if not self.back:
            raise Exception("Can not pop from an empty queue")
        answer = self.back
        self.back = answer.prevNode
        if self.back:
            self.back.nextNode = None
        else:
            self.front = None
        return answer.key

    def is_empty(self):
        return not self.front


if __name__ == '__main__':
    print(p1(k=8))
    print('-----------------------------')
    print(p2_a(x=[], y=[1, 3, 5]))
    print(p2_b(x=[2, 4, 6], y=[]))
    print(p2_c(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print(p2_d(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print('------------------------------')
    print(p3_a(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_b(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_c(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print('------------------------------')
    print(p4_a())
    print(p4_b(p4_a()))
    print('------------------------------')
    graph = {
        'A': ['D', 'E'],
        'B': ['E', 'F'],
        'C': ['E'],
        'D': ['A', 'E'],
        'E': ['A', 'B', 'C', 'D'],
        'F': ['B'],
        'G': []
    }
    print(p5_a(graph))
    print(p5_b(graph))
    print(p5_c(graph))
    print(p5_d(graph))
    print('------------------------------')
    pq = PriorityQueue()
    pq.push('apple')
    pq.push('kiwi')
    pq.push('orange')
    while not pq.is_empty():
        print(pq.pop())
