from typing import Iterable, Tuple, Dict

class GraphNode:
    def __init__(self, nodeID:str, edges: Iterable):
        self.id = nodeID
        self.edges = edges
    
    def __str__(self):
        return f'{self.id}'

    def addEdges(self, edges: Iterable):
        self.edges.extend(edges)

    def neighbors(self):
        return [ edge.target.id for edge in self.edges ]

class GraphEdge:
    def __init__(self, source:GraphNode, target:GraphNode, cost:float = 0):
        self.source = source
        self.target = target
        self.cost = cost
    
    def __str__(self):
        return f'{self.source},{self.target},{self.cost}'


class Graph:
    def __init__(self, nodes: Iterable[str], edges: Iterable[Tuple[str,str,float]]):
        self._nodes = { nodeID: GraphNode(nodeID,[]) for nodeID in nodes }
        edgesBySource = { nodeID: list() for nodeID in self._nodes.keys() }

        for source, target, cost in edges:
            edgesForSource = edgesBySource.get(source, [])
            sourceNode,targetNode = self._nodes[source], self._nodes[target]
            edgesForSource.append(GraphEdge(sourceNode, targetNode, cost))

        for nodeID in nodes:
            node = self._nodes[nodeID]
            edges = edgesBySource[nodeID]
            node.addEdges(edges)

    def edges(self) -> Iterable[GraphEdge]:
        edges = []
        for node in self.nodes().values():
            for edge in node.edges:
                edges.append(edge)
        return edges

    def nodes(self) -> Dict[str, GraphNode]:
        return self._nodes

class SearchAlgorithm:
    @classmethod
    def recover_path(cls, target, start, previous_nodes):
        path = [target]
        node = target
        while node != start:
            node = previous_nodes[node]
            path.append(node)
        return list(reversed(path))
            
    @classmethod
    def Path(cls, graph:Graph, source:str, target: str):
        raise Exception("NotImplemented")


class DFSSearchAlgorithm(SearchAlgorithm):
    def __init__(self):
        pass

    @classmethod
    def Path(cls, graph:Graph, source:str, target: str):
        from collections import deque
        nodes = graph.nodes()
            
        '''returns the path to target, and exploration order'''
        deq: deque = deque()
        previous_nodes: dict = dict()
        explored: list = list()

        deq.append(source)
        previous_nodes[source] = None

        while len(deq) != 0:
            exploring = deq.pop()
            explored.append(exploring)

            if exploring == target:
                path = cls.recover_path(target, source, previous_nodes)
                return path, explored

            node = nodes[exploring]
            children = node.neighbors()
            for child in sorted(children, reverse=True):
                if child not in explored and child not in deq:
                    deq.append(child)
                    previous_nodes[child] = exploring

        return [], explored


class Search:
    def __init__(self, graph:Graph, strategy:SearchAlgorithm):
        self.graph = graph
        self.strategy = strategy

    def Path(self, source:str, target:str):
        return self.strategy.Path(self.graph, source, target)

def main():
    nodes = ['A', 'B']
    edges = [ ('A', 'B', 1) ]
    graph = Graph(nodes, edges)
    graphEdges = graph.edges()
    for edge in graphEdges:
        print(edge)
    
    searchStrategy = DFSSearchAlgorithm()
    search = Search(graph, searchStrategy)
    path, explored = search.Path("A", "B")

    print()
    print()
    print("Path")
    print(path)

    print()
    print()
    print("Explored")
    print(explored)

    '''
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

    nodes = adj_costs.keys()
    edges = []
    for source, node_edges in adj_costs.items():
        for target, cost in node_edges.items():
            edges.append((source, target, cost))
    '''

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
    nodes = adj_list.keys()
    edges = []
    for source, node_edges in adj_list.items():
        for target in node_edges:
            edges.append((source, target, 1))
    
    graph = Graph(nodes, edges)
    graphEdges = graph.edges()
    for edge in graphEdges:
        print(edge)
    
    searchStrategy = DFSSearchAlgorithm()
    search = Search(graph, searchStrategy)
    path, explored = search.Path("A", "F")

    print()
    print()
    print("Path")
    print(path)

    print()
    print()
    print("Explored")
    print(explored)


if __name__ == '__main__':
    main()