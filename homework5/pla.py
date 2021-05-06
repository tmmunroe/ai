import sys
import pandas
import numpy as np
from typing import Callable, List, Tuple
from collections import namedtuple

    
def learn_perceptron(examples: pandas.DataFrame) -> pandas.DataFrame:
    #add b
    loc = len(examples.columns) - 1
    examples.insert(loc, 'b', 1)

    np_arrayed = examples.to_numpy()
    feature_count = len(np_arrayed[0]) - 1

    #initialize weights to 0
    weights = np.array( [ float(0) for f in range(feature_count) ])
    all_weights = np.array([weights])

    while True:
        print("NEXT")
        for i, row in enumerate(np_arrayed):
            features, label = row[:-1], row[-1]
            f_x = weights.dot(features)
            if f_x * label <= 0:
                weights = weights + label*features
            print(f"{i}: {f_x}, {label}")

        if np.array_equal(weights, all_weights[-1]):
            break
        all_weights = np.vstack((all_weights, weights))

    return pandas.DataFrame(all_weights)

def run_learner(fin:str, fout:str):
    print(f"Input: {fin}")
    print(f"Output: {fout}")

    inputs = pandas.read_csv(filepath_or_buffer=fin, header=None)
    outputs = learn_perceptron(inputs)
    outputs.to_csv(path_or_buf=fout, header=False,index=False)
    

    print(f"Done")

def main():
    if len(sys.argv) != 3:
        print(f"Incorrect args.. Program should be invoked with python3 pla.py [input file] [output file]")
        sys.exit(1)
    
    fin, fout = sys.argv[1], sys.argv[2]
    run_learner(fin, fout)

if __name__ == "__main__":
    #main()
    run_learner('/home/tmandevi/artificialIntelligence/homework5/data/data1.csv', '/home/tmandevi/artificialIntelligence/homework5/out_pla.txt')