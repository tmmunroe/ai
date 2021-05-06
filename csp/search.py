
from typing import Optional, Iterable, Tuple, Set, List, Dict, Hashable, Any, Callable, Sequence
import sys
import time
import math
import copy
import csp.csp.CSP as CSP
import csp.constraint.BinaryConstraint as BinaryConstraint

def forward_check(assignment:Tuple[Hashable,Any], domains:Dict, csp:CSP) -> Tuple[bool, Dict]:
    """takes in assignments and applies forward checking to the variable domains
    according to the given constraints
    returns a tuple indicating if the forward check was consistent and the new domains it generated"""
    new_domains = dict(domains)
    var,val = assignment
    constraints_for_var = csp.forVariable(var)
    for constraint in constraints_for_var:
        other = constraint.other(var)
        domain = domains.get(other, None)
        if not domain:
            continue

        fc_domain = { value for value in domain if constraint.check(assignment, (other,value)) }
        if len(fc_domain) == 0:
            return False, {}
        
        new_domains[other] = fc_domain

    return True, new_domains

def pop_minimum_remaining_value(domains:Dict) -> Tuple[str, Iterable]:
    def domain_size(key) -> int:
        return len(domains[key])

    minEntry = min(domains.keys(), key=domain_size)
    return minEntry, domains.pop(minEntry)


def order_domain_values(variable, domain:Iterable, domains:Dict, csp:CSP) -> Iterable:
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


def goal_test(assignments:Dict, variables:Tuple, csp:CSP) -> bool:
    if len(assignments) != len(variables):
        return False
    return csp.checkAssignments(assignments)


def backtracking_recursive(variables:Tuple, assignments:Dict, domains:Dict, csp:CSP) -> Optional[Dict]:
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


def trim_assigned_domains(domains:Dict, assigments:Dict) -> Dict:
    new_domains = copy.deepcopy(domains)
    for assigned in assigments:
        del new_domains[assigned]
    return new_domains


def backtracking_general(variables:Tuple, assignments:Dict, domains:Dict, constraints:List[BinaryConstraint]) -> Optional[Dict]:
    '''
    Inputs-
    variables [List]: all variables in the state
    assignments [Dict]: all assignments in the initial state
    domains [Dict]: the domains for each variable
    constraints [List]: all constraints
    
    Outputs-
    results [Dict]: all assignments in the final state of the game, or None if a solution could not be found
    '''
    csp: CSP = CSP(variables, constraints)
    all_variables = tuple(variables)

    unassigned_domains = trim_assigned_domains(domains, assignments)
    for assigment in assignments.items():
        ok, new_domains = forward_check(assigment, unassigned_domains, csp)
        if not ok:
            raise Exception("Unsolvable board???")
        unassigned_domains = new_domains

    return backtracking_recursive(all_variables, assignments, unassigned_domains, csp)