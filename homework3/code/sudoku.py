#!/usr/bin/env python
#coding:utf-8

from typing import Optional, Iterable, Tuple, Set, List, Dict, Hashable, Any, Callable, Sequence
from collections import UserDict
import sys
import time

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

class BinaryConstraint:
    def __init__(self, first:Hashable, second:Hashable, testFunc:Callable):
        self.first = first
        self.second = second
        self.testFunc = testFunc
    
    def __contains__(self, member:Hashable):
        return (member == self.first) or (member == self.second)
    
    def other(self, member:Hashable):
        if member == self.first:
            return self.second
        return self.first

    def check(self, assignmentA:Tuple[Hashable,Any], assignmentB:Tuple[Hashable,Any]) -> bool:
        firstVal = secondVal = None
        if (assignmentA[0] == self.first) and (assignmentB[0] == self.second):
            firstVal, secondVal = assignmentA[1], assignmentB[1]
        elif (assignmentA[0] == self.second) and (assignmentB[0] == self.first):
            firstVal, secondVal = assignmentB[1], assignmentA[1]
        else:
            raise Exception(f"Invalid inputs checking {self}: {assignmentA}, {assignmentB}")

        return self.testFunc(firstVal, secondVal)

    def checkAssignments(self, assignments:Dict) -> bool:
        if (self.first not in assignments) or (self.second not in assignments):
            return False
        return self.testFunc(assignments[self.first], assignments[self.second])


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


def generateInitialState(board:Dict) -> Tuple[Iterable,Dict,Dict]:
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
    return variables, assignments, domains

def forward_check(assignment:Tuple[Hashable,Any], domains:Dict, csp:Iterable[BinaryConstraint]) -> Tuple[bool, Dict]:
    """takes in assignments and applies forward checking to the variable domains
    according to the given constraints
    returns a tuple indicating if the forward check was consistent and the new domains it generated"""
    new_domains = dict(domains)
    var,val = assignment
    constraints_for_var = ( c for c in csp if var in c )
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


def check_constraints(assignments:Dict, csp:Iterable[BinaryConstraint]) -> bool:
    return all((constraint.checkAssignments(assignments) for constraint in csp))

def pop_minimum_remaining_value(domains:Dict) -> Tuple[str, Iterable]:
    def domain_size(key) -> int:
        return len(domains[key])

    minEntry = min(domains.keys(), key=domain_size)
    return minEntry, domains.pop(minEntry)

def order_domain_values(variable, domain:Iterable, domains:Dict, csp:Iterable[BinaryConstraint]) -> Iterable:
    fc_domains = []

    for value in domain:
        ok, new_domains = forward_check((variable, value), domains, csp)
        if not ok:
            continue
        total_domain_size = sum((len(d) for d in new_domains.values()))
        fc_domains.append((value, new_domains, total_domain_size))
    
    def domain_size(tup) -> int:
        return tup[2]

    return sorted(fc_domains, key=domain_size, reverse=True)

def goal_test(assignments:Dict, variables:Tuple, csp:Iterable[BinaryConstraint]) -> bool:
    if len(assignments) != len(variables):
        return False
    return check_constraints(assignments, csp)


def backtracking_recursive(variables:Tuple, assignments:Dict, domains:Dict, csp:Iterable[BinaryConstraint]) -> Optional[Dict]:
    """returns None if no solution was found or a solved board"""
    if goal_test(assignments, variables, csp):
        return assignments

    var, values = pop_minimum_remaining_value(domains)
    
    '''
    ordered_values = order_domain_values(var, values, domains, csp)
    for value, fc_domains, _ in ordered_values:
    '''
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

def backtracking(board:Dict):
    """Takes a board and returns solved board."""
    variables, assignments, domains = generateInitialState(board)
    csp: List[BinaryConstraint] = generateConstraints()

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

    print(f'Solved {len( [bs for bs in allBs if bs.solved] )} of {len(allBs)} Sudokus')
    print(f'Total runtime: {sum((bs.runTime() for bs in allBs))}')
    print(f'Mean runtime: {sum((bs.runTime() for bs in allBs)) / len(allBs)}')
    print(f'Max runtime: {max((bs.runTime() for bs in allBs))}')
    print(f'Min runtime: {min((bs.runTime() for bs in allBs))}')
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
    
    
    solvedBoard, bs = solve_line_board('800000000003600000070090200050007000000045700000100030001000068008500010090000400')
    if solvedBoard != '812753649943682175675491283154237896369845721287169534521974368438526917796318452':
        print('UGH.. no good')
    else:
        print('Success!!')