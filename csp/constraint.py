from typing import Optional, Iterable, Tuple, Set, List, Dict, Hashable, Any, Callable, Sequence
import sys
import time
import math

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
