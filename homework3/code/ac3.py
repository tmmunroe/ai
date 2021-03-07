

def all_arcs(var_neighbor_constraints:Dict[str, Dict]) -> Set:
    arcs = set()
    for var, neigbor_constraints in var_neighbor_constraints.items():
        for neighbor in neigbor_constraints:
            arcs.add((var, neighbor))
    return arcs

def neighbor_constraint_by_var(domains:Dict, csp:CSP) -> Dict[str, Dict]:
    neighbor_constraint: Dict[str, Dict] = {}
    for x in domains:
        x_neighbor_constraints = neighbor_constraint.get(x, dict())
        constraints_for_var = csp.forVariable(x)
        for constraint in constraints_for_var:
            other = constraint.other(x)
            if other in domains:
                x_neighbor_constraints[other] = constraint
    return neighbor_constraint

def revise(start, start_domain:Iterable, end, end_domain:Iterable, constraint:BinaryConstraint) -> Tuple[bool, Set]:
    """takes in a start variable, start domain, end variable, end domain, and the constraint for the
    variables
    returns a tuple indicating if there was a revision in the start domain and the new domain if there was"""
    end_assignments = [ (end,value) for value in end_domain ]
    revised_start_domain = set(start_domain)
    revised = False
    for value in revised_start_domain:
        s = start,value
        if not any((constraint.check(s,e) for e in end_assignments)):
            revised_start_domain.remove(value)
            revised = True
    return revised, revised_start_domain


def arc_consistency_check(domains:Dict, csp:CSP) -> Tuple[bool, Dict]:
    """takes in assignments and applies forward checking to the variable domains
    according to the given constraints
    returns a tuple indicating if the forward check was consistent and the new domains it generated"""
    new_domains = dict(domains)
    neighbor_constraints: Dict[str, Dict] = neighbor_constraint_by_var(domains, csp)
    arcs_to_check = deque( all_arcs(neighbor_constraints) )

    while len(arcs_to_check) != 0:
        start, end = arcs_to_check.popleft()
        revised, ac_start_domain = revise(start, new_domains[start], end, new_domains[end], neighbor_constraints[start][end])
        if revised:
            if len(ac_start_domain) == 0:
                return False, {}
            neighbors_to_add = (neighbor for neighbor in neighbor_constraints[start] if neighbor != end)
            for neighbor in neighbors_to_add:
                arcs_to_check.append((neighbor, start))
            new_domains[start] = ac_start_domain

    return True, new_domains
