#!/usr/bin/env python
#coding:utf-8

from typing import Optional, Iterable, Tuple, Set, List, Dict, Hashable, Any, Callable, Sequence
import sys
import time
import math
import copy
from csp import CSP
from constraint import BinaryConstraint
import search

"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8
"""

ROW = "ABCDEFGHI"
COL = "123456789"

class BoardStats:
    def __init__(self, board:str):
        self.board = board
        self.solved = False
        self.finalBoard = None
        self.startTime = self.endTime = None
    
    def start(self):
        self.startTime = time.time()

    def stop(self):
        self.endTime = time.time()
    
    def runTime(self):
        return self.endTime - self.startTime
    
    def solution(self, finalBoard):
        self.finalBoard = finalBoard
        self.solved = True


def print_board(board:Dict):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print(row)


def board_to_string(board:Dict):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)

def notEqual(a:Any,b:Any) -> bool:
    return a != b

def orderPair(itemA:Any, itemB:Any) -> Tuple[Any,Any]:
    if itemB > itemA:
        return itemA, itemB
    return itemB, itemA

def orderedPairs(unique_elements:Sequence[Any]):
    pairs = []
    for index, element in enumerate(unique_elements):
        for other_element in unique_elements[index+1:]:
            pairs.append(orderPair(element, other_element))
    return pairs

def grids():
    generated_grids = []
    grid_dim = 3
    rowBlocks = [ROW[rowBlock:rowBlock+grid_dim] for rowBlock in range(0,9,grid_dim)]
    colBlocks = [COL[colBlock:colBlock+grid_dim] for colBlock in range(0,9,grid_dim)]
    for rows in rowBlocks:
        for cols in colBlocks:
            grid = [f'{row}{col}' for row in rows for col in cols]
            generated_grids.append(grid)
    return generated_grids

def generateConstraints():
    uniqueArcs: Set[Tuple[str,str]] = set()
    for row in ROW:
        members = [ f"{row}{col}" for col in COL ]
        pairs = orderedPairs(members)
        for pair in pairs:
            uniqueArcs.add(pair)
    
    for col in COL:
        members = [ f"{row}{col}" for row in ROW ]
        pairs = orderedPairs(members)
        for pair in pairs:
            uniqueArcs.add(pair)

    for grid in grids():
        pairs = orderedPairs(grid)
        for pair in pairs:
            uniqueArcs.add(pair)
    
    return [ BinaryConstraint(first, second, notEqual) for first,second in uniqueArcs ]

def generateInitialState(board:Dict) -> Tuple[Tuple,Dict,Dict]:
    variables = []
    assignments = {}
    domains = {}
    initial_domains = range(1,10)
    for cell, value in board.items():
        variables.append(cell)
        if value != 0:
            assignments[cell] = value
        else:
            domains[cell] = set(initial_domains)
    return tuple(variables), assignments, domains

def backtracking(board:Dict):
    """Takes a board and returns solved board."""
    variables, assignments, domains = generateInitialState(board)
    constraints =  generateConstraints()
    return search.backtracking_general(variables, assignments, domains, constraints)

def solve_line_board(line:str) -> Tuple[str, BoardStats]:
    # Parse boards to dict representation, scanning board L to R, Up to Down
    board = { ROW[r] + COL[c]: int(line[9*r+c])
                for r in range(9) for c in range(9)}

    # Print starting board. TODO: Comment this out when timing runs.
    #print_board(board)

    # Solve with backtracking
    bs = BoardStats(line)
    bs.start()
    solved_board = backtracking(board)
    bs.stop()
    bs.solution(solved_board)

    # Print solved board. TODO: Comment this out when timing runs.
    #print_board(solved_board)

    return board_to_string(solved_board), bs

def batch_file():
    #  Read boards from source.
    src_filename = 'sudokus_start.txt'
    try:
        srcfile = open(src_filename, "r")
        sudoku_list = srcfile.read()
    except:
        print("Error reading the sudoku file %s" % src_filename)
        exit()

    allBs = []

    # Setup output file
    out_filename = 'output.txt'
    outfile = open(out_filename, "w")

    # Solve each board using backtracking
    for index, line in enumerate(sudoku_list.split("\n")):
        #print(f"Solving board {index}: {line}")
        if index % 50 == 0:
            print(f'Solving board {index}')
        if len(line) < 9:
            continue
        
        solved_board, board_stats = solve_line_board(line)
        allBs.append(board_stats)

        # Write board to file
        outfile.write(solved_board)
        outfile.write('\n')

    with open('stats.txt', 'w') as f:
        totalTime = sum((bs.runTime() for bs in allBs))
        mean = totalTime / len(allBs)
        variance = sum(((bs.runTime() - mean)**2 for bs in allBs)) / len(allBs)
        print(f'Solved: {len( [bs for bs in allBs if bs.solved] )} / {len(allBs)}', file=f)
        #print(f'Total runtime: {totalTime}', file=f)
        print(f'Mean runtime: {mean}', file=f)
        print(f'Standard Deviation: {math.sqrt(variance)}', file=f)
        print(f'Max runtime: {max((bs.runTime() for bs in allBs))}', file=f)
        print(f'Min runtime: {min((bs.runTime() for bs in allBs))}', file=f)

    print("Finishing all boards in file.")

def test_batch_file():
    batch_file()
    allActual = allExpected = None
    with open('output.txt') as f:
        allActual = [ line for line in f ]
    with open('sudokus_finish.txt') as f:
        allExpected = [ line for line in f ]

    for line, (actual, expected) in enumerate(zip(allActual,allExpected)):
        if actual != expected:
            print(f'Failed {line}:\nExpected: {expected}\nGot: {actual}\n')
            for c, (a,e) in enumerate(zip(actual,expected)):
                print(f'{c}: {a} - {e}')
                if a != e:
                    raise Exception(f'Error at character {c} {a} vs {e}')

    print('All boards solved correctly')

if __name__ == '__main__':
    '''
    test_batch_file()
    
    solvedBoard, bs = solve_line_board('800000000003600000070090200050007000000045700000100030001000068008500010090000400')
    if solvedBoard != '812753649943682175675491283154237896369845721287169534521974368438526917796318452':
        print('UGH.. no good')
    else:
        print('Success!!')
    '''
    
    if len(sys.argv) == 1:
        test_batch_file()
    elif len(sys.argv) == 2:
        board = sys.argv[1]
        solution, _ = solve_line_board(board)
        with open('output.txt', 'w') as f:
            print(solution, file=f)
    elif len(sys.argv) == 3:
        board = sys.argv[1]
        expected = sys.argv[2]
        solution, _ = solve_line_board(board)
        print(f"Solved: {solution==expected}")    
