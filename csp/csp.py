
from typing import Optional, Iterable, Tuple, Set, List, Dict, Hashable, Any, Callable, Sequence
import sys
import time
import math
from constraint import BinaryConstraint

class CSP:
    def __init__(self, variables:Iterable[Hashable], constraints:Iterable[BinaryConstraint]):
        self.variables = list(variables)
        self.constraints = list(constraints)
        self.varConstraints = { var: [c for c in constraints if var in c] for var in variables }
    
    def forVariable(self, variable:Hashable) -> Iterable[BinaryConstraint]:
        if variable in self.varConstraints:
            return self.varConstraints[variable]
        return []
    
    def checkAssignments(self, assignments:Dict):
        return all((constraint.checkAssignments(assignments) for constraint in self.constraints))