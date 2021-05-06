from typing import Callable, List, Tuple
from collections import namedtuple
import sklearn
import matplotlib

import sys
import pandas
import numpy as np
import numpy.typing
import matplotlib.pyplot
import sklearn.cluster

def learn_clusters(image:numpy.typing.ArrayLike, clusters:int) -> sklearn.cluster.KMeans:
    kmeans = sklearn.cluster.KMeans(n_clusters=clusters).fit(image)
    print(kmeans.labels_)
    print(kmeans.cluster_centers_)
    return kmeans

def plot(kmeans: sklearn.cluster.KMeans, x:int, y:int, z:int, cluster_size:int):
    fig = matplotlib.pyplot.figure(figsize = (15,8))
    image_new = kmeans.cluster_centers_[kmeans.labels_].reshape(x, y, z)
    matplotlib.pyplot.imsave(f'fig_clustering_{cluster_size}.png', image_new)
    matplotlib.pyplot.close(fig=fig)

def run_learner(fin:str, fout:str):
    print(f"Input: {fin}")
    print(f"Output: {fout}")

    image = matplotlib.pyplot.imread(fin)
    x,y,z = image.shape
    image_new = image.reshape(x*y,z)
    for clusters in range(1,22):
        print(f"Clustering: {clusters}")
        results = learn_clusters(image_new, clusters)
        plot(results, x, y, z, clusters)

def main():
    if len(sys.argv) != 3:
        print(f"Incorrect args.. Program should be invoked with python3 pla.py [input file] [output file]")
        sys.exit(1)
    
    fin, fout = sys.argv[1], sys.argv[2]
    run_learner(fin, fout)

if __name__ == "__main__":
    #main()
    run_learner('/home/tmandevi/artificialIntelligence/homework5/data/trees.png', '/home/tmandevi/artificialIntelligence/homework5/out_clustering.txt')