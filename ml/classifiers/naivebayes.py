"""
TODO: add m-count probabilities to handle 0s
"""

from collections import namedtuple, Counter, UserList
from typing import Any, Dict, List, Iterable, Sequence

def to_dict(sample) -> Dict:
    if isinstance(sample, Sequence):
        return {i:value for i,value in enumerate(sample)}
    elif isinstance(sample, Dict):
        return sample
    raise TypeError(f"Can not handle type {type(sample)}")

class AttributeDistribution:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.distribution = Counter()
    
    def count(self, value=None):
        return self.distribution[value]

    def probability(self, value) -> float:
        return self.count(value) / self.total
    
    def add(self, value) -> None:
        self.total += 1
        self.distribution[value] += 1

class ClassModel:
    def __init__(self, name):
        self.name = name
        self.sampleCount = 0
        self.attribute_distributions: Dict = {}
    
    def attributeDistribution(self, key) -> AttributeDistribution:
        if key not in self.attribute_distributions:
            self.attribute_distributions[key] = AttributeDistribution(key)
        return self.attribute_distributions[key]

    def add_sample(self, sample:Dict):
        self.sampleCount += 1
        for key, value in sample.items():
            attributeDist = self.attributeDistribution(key)
            attributeDist.add(value)
    
    def probability(self, sample:Dict) -> float:
        total: float = 1
        for key, value in sample.items():
            attribute = self.attributeDistribution(key)
            total *= attribute.probability(value)
        return total

class NaiveBayes:
    def __init__(self):
        self.class_models = {}
        self.sampleCount = 0
    
    def class_model(self, label) -> ClassModel:
        if label not in self.class_models:
            self.class_models[label] = ClassModel(label)
        return self.class_models[label]

    def probability(self, sample:Dict, label) -> float:
        model = self.class_model(label)
        label_probability = model.sampleCount / self.sampleCount
        sample_probability = model.probability(sample)
        print(f"{label} {sample} Probability: {label_probability}, {sample_probability}")
        return label_probability * sample_probability

    def add_sample(self, sample:Dict, label) -> None:
        self.sampleCount += 1
        model = self.class_model(label)
        model.add_sample(sample)

    def __call__(self, sample) -> Any:
        sample_dict = to_dict(sample)
        best, best_probability = None, -1.
        for label in self.class_models:
            model_probability = self.probability(sample_dict, label)
            if model_probability > best_probability:
                best, best_probability = label, model_probability
        return best

    def loss(self, labels:Iterable, predictions:Iterable) -> float:
        totalLoss = 0
        for label, prediction in zip(labels, predictions):
            if label != prediction:
                totalLoss += 1
        return totalLoss

    def train(self, data:Iterable, labels:Iterable) -> Any:
        sample_dicts = [ to_dict(sample) for sample in data ]
        for sample, label in zip(sample_dicts, labels):
            self.add_sample(sample, label)
        predictions = (self(sample) for sample in data)
        return self.loss(labels, predictions)

    def test(self, samples:Iterable, labels:Iterable) -> Any:
        predictions = ( self(sample) for sample in samples )
        return self.loss(labels, predictions)