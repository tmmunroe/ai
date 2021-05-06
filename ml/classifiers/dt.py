from typing import Any, Callable, Dict, Hashable, List, Tuple
import statistics
from collections import namedtuple, Counter, UserList
from math import log2

def entropy(labels: List) -> float:
    total = len(labels)
    counts = Counter(labels)
    proportions = (count / total for count in counts.values())
    terms = (p*log2(p) for p in proportions if p != 0)
    return -sum(terms)

def gini(labels: List) -> float:
    total = len(labels)
    counts = Counter(labels)
    if len(counts.items()) > 2:
        raise Exception("Too many labels for Gini.. must be binary")
    proportions = (count / total for count in counts.values())
    terms = (p*p for p in proportions)
    gi = sum(terms)
    return 1 - gi


class Sample(namedtuple('Sample', ['attributes', 'label'])):
    def attribute(self, index:int) -> Any:
        return self.attributes[index]


class Population(UserList):
    def majority(self):
        labels = [ s.label for s in self ]
        return statistics.mode(labels)

    def entropy(self):
        labels = [ s.label for s in self ]
        return entropy(labels)
    
    def gini(self):
        labels = [ s.label for s in self ]
        return gini(labels)
    
    def homogenousAttributes(self):
        if len(self) == 0:
            return True
        first = self[0]
        return all((s.attributes == first.attributes for s in self))

def splitPopulation(population:Population, feature:int) -> Dict[Hashable, Population]:
    subpopulations: Dict[Hashable, Population] = {}
    for sample in population:
        value = sample.attribute(feature)
        if value in subpopulations:
            subpopulations[value].append(sample)
            continue
        subpopulations[value] = Population([sample])
    return subpopulations


class DecisionTree:
    def __init__(self):
        self.feature:int = -1
        self.subtrees: Dict[Hashable, DecisionTree] = {}
        self.isLeaf = False
        self.population: Population = Population()
        self.maxEntropy = 0.1

    def gain(self, feature:int) -> float:
        root_entropy = self.population.entropy()
        root_population = float(len(self.population))
        
        subpopulations = splitPopulation(self.population, feature)
        children_weights = ( len(subpop)/root_population for subpop in subpopulations.values() )
        children_entropy = ( subpop.entropy() for subpop in subpopulations.values() )
        children_weighted_entropy = sum((w*e for w,e in zip(children_weights, children_entropy)))

        return root_entropy - children_weighted_entropy

    def __call__(self, instance:List) -> Any:
        #if this is a leaf, return the majority label at this leaf
        if self.isLeaf:
            return self.population.majority()
        #else get feature value and descend in that values subtree
        feature_value = instance[self.feature]
        subtree = self.subtrees[feature_value]
        return subtree(instance)

    def loss(self, labels:List, predictions:List) -> float:
        totalLoss = 0
        for label, prediction in zip(labels, predictions):
            if label != prediction:
                totalLoss += 1
        return totalLoss

    def train_on_population(self, population:Population) -> Any:
        self.population = population
        print(f"\n\nTraining on population: {population}")

        #base case: if entropy is sufficient with the given labels or population is homogenous, set as leaf node and return error
        sample:Sample = self.population[0]
        if self.population.entropy() <= self.maxEntropy or self.population.homogenousAttributes():
            self.isLeaf = True
            print("Sufficient split")
            labels = [ sample.label for sample in self.population ]
            predictions = [ self(sample.attributes) for sample in self.population ]
            return self.loss(labels, predictions)

        #find feature that has the best information gain
        features:int = len(sample.attributes)
        self.feature = max(range(features), key=self.gain)
        print(f"Splitting on feature {self.feature}")

        #split data and labels into subpopulations based on that feature
        subpopulations = splitPopulation(self.population,self.feature)
        print(f"Subpopulations: {subpopulations}")

        #recursive case: train subtrees for each split
        training_error = 0
        for feature_value, subpop in subpopulations.items():
            print(f"  FeatureValue: {feature_value} Subpopulation: {subpop}")
            subtree = DecisionTree()
            training_error += subtree.train_on_population(subpop)
            self.subtrees[feature_value] = subtree

        return training_error

    def train(self, data:List, labels:List) -> Any:
        population = Population([ Sample(d,l) for d,l in zip(data, labels) ])
        return self.train_on_population(population)

    def test(self, samples:List, labels:List) -> Any:
        predictions = [ self(sample) for sample in samples ]
        return self.loss(labels, predictions)