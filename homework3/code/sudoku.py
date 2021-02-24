#!/usr/bin/env python
#coding:utf-8

from typing import Optional, Iterable, Tuple, Set, List, Mapping
from collections import UserDict
import sys
import math
import time

"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8
"""

ROW = "ABCDEFGHI"
COL = "123456789"

class BoardStats:
    def __init__(self, board):
        self.board = board
        self.startTime = self.endTime = None
    
    def start(self):
        self.startTime = time.time()

    def stop(self):
        self.endTime = time.time()
    
    def runTime(self):
        return self.endTime - self.startTime

class BinaryCSP(list):
    pass

class VariableDomains(dict):
    pass

class BinaryConstraint:
    def __init__(self, first, second, testFunc):
        self.first = first
        self.second = second
        self.testFunc = testFunc
    
    def __contains__(self, member):
        return (member == self.first) or (member == self.second)
    
    def other(self, member):
        if member == self.first:
            return self.second
        return self.first

    def check(self, assignmentA:tuple, assignmentB:tuple) -> bool:
        firstVal = secondVal = None
        if (assignmentA[0] == self.first) and (assignmentB[0] == self.second):
            firstVal, secondVal = assignmentA[1], assignmentB[1]
        elif (assignmentA[0] == self.second) and (assignmentB[0] == self.first):
            firstVal, secondVal = assignmentB[1], assignmentA[1]
        else:
            raise Exception(f"Invalid inputs checking {self}: {assignmentA}, {assignmentB}")

        return self.testFunc(firstVal, secondVal)

    def checkAssignments(self, assignments:dict) -> bool:
        if (self.first not in assignments) or (self.second not in assignments):
            return False
        return self.testFunc(assignments[self.first], assignments[self.second])


def print_board(board):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print(row)


def board_to_string(board):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)


def orderPair(itemA, itemB):
    if itemB > itemA:
        return itemA, itemB
    return itemB, itemA

def orderedPairs(unique_elements):
    pairs = []
    for index, element in enumerate(unique_elements):
        for other_element in unique_elements[index+1:]:
            pairs.append(orderPair(element, other_element))
    return pairs

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

def prepareInitialState(board:dict) -> Tuple[Iterable,Mapping,Mapping]:
    board_dimension = int(math.sqrt(len(board)))
    variables = []
    assignments = {}
    domains = {}
    initial_domains = range(1,board_dimension+1)
    for cell, value in board.items():
        variables.append(cell)
        if value != 0:
            assignments[cell] = value
        else:
            domains[cell] = set(initial_domains)
    return variables, assignments, domains

def notEqual(a,b) -> bool:
    return a != b

def forward_check(assignment, domains:dict, csp:Iterable[BinaryConstraint]) -> Tuple[bool, dict]:
    """takes in assignments and applies forward checking to the variable domains
    according to the given constraints
    returns a tuple indicating if the forward check was consistent and the new domains it generated"""
    new_domains = dict(domains)
    var,val = assignment
    constraints_for_var = [ c for c in csp if var in c ]
    for constraint in constraints_for_var:
        other = constraint.other(var)
        domain = domains.get(other, None)
        if not domain:
            continue

        fc_domain = [ value for value in domain if constraint.check(assignment, (other,value)) ]
        if len(fc_domain) == 0:
            return False, {}
        
        new_domains[other] = fc_domain

    return True, new_domains


def check_constraints(assignments:dict, csp:Iterable[BinaryConstraint]) -> bool:
    return all((constraint.checkAssignments(assignments) for constraint in csp))

def pop_minimum_remaining_value(domains:dict) -> Tuple[str, Iterable]:
    def domain_size(key) -> int:
        return len(domains[key])

    minEntry = min(domains.keys(), key=domain_size)
    return minEntry, domains.pop(minEntry)

def goal_test(assignments:dict, variables:tuple, csp:Iterable[BinaryConstraint]) -> bool:
    if len(assignments) != len(variables):
        return False
    return check_constraints(assignments, csp)


def backtracking_recursive(variables:tuple, assignments:dict, domains:dict, csp:Iterable[BinaryConstraint]) -> Optional[dict]:
    """returns None if no solution was found or a solved board"""
    if goal_test(assignments, variables, csp):
        return assignments

    var, values = pop_minimum_remaining_value(domains)
    for value in values:
        ok, fc_domains = forward_check((var,value), domains, csp)
        if not ok:
            continue

        assignments[var] = value
        result = backtracking_recursive(variables, assignments, fc_domains, csp)
        if result:
            return result
        del assignments[var]

    domains[var] = values
    return None

def backtracking(board:dict):
    """Takes a board and returns solved board."""
    assignments = {}
    domains = {}
    variables = []
    initial_domains = range(1,10)
    csp: List[BinaryConstraint] = generateConstraints()

    for cell, value in board.items():
        variables.append(cell)
        if value != 0:
            assignments[cell] = value
        else:
            domains[cell] = set(initial_domains)

    for assigment in assignments.items():
        ok, new_domains = forward_check(assigment, domains, csp)
        if not ok:
            raise Exception("Unsolvable board???")
        domains = new_domains

    #print(f"Variables: {len(variables)}")
    #print(f"Assignments: {len(assignments)}")
    #print(f"Domains: {len(domains)}")
    solved_board = backtracking_recursive(tuple(variables), assignments, domains, csp)
    return solved_board

def solve_line_board(line):
    # Parse boards to dict representation, scanning board L to R, Up to Down
    board = { ROW[r] + COL[c]: int(line[9*r+c])
                for r in range(9) for c in range(9)}

    # Print starting board. TODO: Comment this out when timing runs.
    #print_board(board)

    # Solve with backtracking
    solved_board = backtracking(board)

    # Print solved board. TODO: Comment this out when timing runs.
    #print_board(solved_board)

    return board_to_string(solved_board)

def batch_file():
    #  Read boards from source.
    src_filename = 'sudokus_start.txt'
    try:
        srcfile = open(src_filename, "r")
        sudoku_list = srcfile.read()
    except:
        print("Error reading the sudoku file %s" % src_filename)
        exit()

    # Setup output file
    out_filename = 'output.txt'
    outfile = open(out_filename, "w")

    # Solve each board using backtracking
    for index, line in enumerate(sudoku_list.split("\n")):
        print(f"Solving board {index}: {line}")
        if len(line) < 9:
            continue

        solved_board = solve_line_board(line)

        # Write board to file
        outfile.write(solved_board)
        outfile.write('\n')

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

if __name__ == '__main__':
    '''
    if len(sys.argv) == 1:
        test_batch_file()
    else:
        board = sys.argv[1]
        solve_line_board(board)
    '''
    test_batch_file()
    
    
    solvedBoard = solve_line_board('800000000003600000070090200050007000000045700000100030001000068008500010090000400')
    if solvedBoard != '812753649943682175675491283154237896369845721287169534521974368438526917796318452':
        print('UGH.. no good')
    else:
        print('Success!!')