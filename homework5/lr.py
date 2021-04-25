import sys
import pandas
import numpy as np
from typing import Callable, List, Tuple
from collections import namedtuple


def normalize_features(examples: pandas.DataFrame) -> pandas.DataFrame:
    normalized_features = examples.copy(deep=True)
    stddevs = normalized_features.std(ddof=0)
    means = normalized_features.mean()
    for col in normalized_features.columns:
        normalized_features[col] = (normalized_features[col] - means[col]) / stddevs[col]
    return normalized_features if True else examples
    
def learn_lr(examples_in: pandas.DataFrame, learning_rate:float, iters:int) -> pandas.DataFrame:
    #prepare data
    features_raw = examples_in.iloc[:,:-1]

    labels_df = examples_in.iloc[:,-1]
    features_df = normalize_features(features_raw)
    features_df.insert(0, 'bias', 1)

    features = features_df.to_numpy()
    labels = labels_df.to_numpy()

    #initialize weights to 0
    population_size = labels.size
    weights = np.zeros(features[0].shape)
    all_weights = np.array([weights])
    f_xs = np.zeros(labels.shape)

    for i in range(iters):
        f_xs = np.matmul(features, weights)
        residuals = f_xs - labels
        resid = sum((abs(r) for r in residuals))
        for j,w in enumerate(weights):
            
            xs_j = features[:,j]
            residual_j = residuals.dot(xs_j)
            weights[j] = w - (learning_rate / population_size) * residual_j
        
        if resid < 4.86:
            print(f"Residual {resid}, Rate: {learning_rate}, Iter: {i} weights: {weights}")
        if np.array_equal(weights, all_weights[-1]):
            break
        all_weights = np.vstack((all_weights, weights))

    out = pandas.DataFrame([weights])
    out.insert(0,'iters', iters)
    out.insert(0,'alpha', learning_rate)
    return out

def run_learner(fin:str, fout:str):
    print(f"Input: {fin}")
    print(f"Output: {fout}")
    parameters = [
        (0.001, 100), 
        (0.005, 100), 
        (0.01, 100), 
        (0.05, 100),
        (0.1, 100),
        (0.5, 100),
        (1., 100),
        (5., 100),
        (10., 100)
    ]

    inputs = pandas.read_csv(filepath_or_buffer=fin, header=None)
    results = []
    for alpha, iters in parameters:
        outs = learn_lr(inputs, alpha, iters)
        results.append(outs)
    outputs = pandas.concat(results)
    outputs.to_csv(path_or_buf=fout, header=False,index=False)

def main():
    if len(sys.argv) != 3:
        print(f"Incorrect args.. Program should be invoked with python3 pla.py [input file] [output file]")
        sys.exit(1)
    
    fin, fout = sys.argv[1], sys.argv[2]
    run_learner(fin, fout)

if __name__ == "__main__":
    #main()
    run_learner('/home/tmandevi/artificialIntelligence/homework5/data/data2.csv', '/home/tmandevi/artificialIntelligence/homework5/out_lr.txt')