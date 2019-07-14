import pdb
import time
import networkx as nx
import numpy as np
import graphwave
from graphwave.graphwave import graphwave_alg
import argparse
from scipy import spatial
from sklearn.metrics import roc_auc_score
from matrix.signal_matrix import generate_matrix
import pickle
import os
from web.analogy import *
from web.evaluate import evaluate_analogy, evaluate_on_semeval_2012_2

parser = argparse.ArgumentParser()
parser.add_argument("--vocabulary_size", type=int, default=5000)
parser.add_argument("--min_count", type=int, default=100)
parser.add_argument("--neg_samples", type=int, default=5)
parser.add_argument("--skip_window", type=int, default=1)
args = parser.parse_args()

if not os.path.isfile("text8_{}.pkl".format(args.vocabulary_size)):
    pmi, dd, rd, nij = generate_matrix(
        "data/word2vec/text8.zip", args.vocabulary_size, args.min_count, args.neg_samples, args.skip_window)
else:
    pmi, dd, rd, nij = pickle.load("text8_{}.pkl".format(args.vocabulary_size))

np.random.seed(args.seed)

print("Reading graph...")
G = nx.from_numpy_array(pmi)
print("Done!")
print("Calculate embedding...")
stime = time.time()
chi, heat_print, taus = graphwave_alg(
    G, np.linspace(0, 100, 25), taus='auto', verbose=True)
pdb.set_trace()
print("Done! Take {}s", time.time()-stime)
outputfile = "size{}_count{}_neg{}_skip{}.npy"
np.save(outputfile, chi)

w2v = {}
for i in range(chi.shape[0]):
    w2v[rd[i]] = chi[i]

analogy_tasks = {
    "Google": fetch_google_analogy(),
    "MSR": fetch_msr_analogy()
}

analogy_results = {}

for name, data in analogy_tasks.items():
    print(name)
    analogy_results[name] = evaluate_analogy(w2v, data.X, data.y)
    logger.info("Analogy prediction accuracy on {} {}".format(
        name, analogy_results[name]))

analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2(w2v)['all']

print(analogy_results)
