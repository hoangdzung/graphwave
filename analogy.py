import networkx as nx
import numpy as np
import graphwave
from graphwave.graphwave import graphwave_alg
import argparse
from scipy import spatial
from sklearn.metrics import roc_auc_score

def sim_vec(embedding, x, cosin=True):
    linearSize = int(len(x)/2)
    vec1 = None
    vec2 = None
    for i in range(linearSize):
        if vec1 is None:
            vec1 = embedding[x[i]]
        else:
            vec1 += embedding[x[i]]

    for i in range(linearSize, len(x)):
        if vec2 is None:
            vec2 = embedding[x[i]]
        else:
            vec2 += embedding[x[i]]
    if cosin:
        return 1 - spatial.distance.cosine(vec1, vec2)
    else:
        return -np.linalg.norm(vec1-vec2)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=97)
parser.add_argument("--combination_file")
parser.add_argument("--edgelist_path")
args = parser.parse_args()

np.random.seed(args.seed)

print("Reading graph...")
G = nx.read_edgelist(args.edgelist_path, nodetype=int)
combinations = np.load(args.combination_file)
linearSize = int(combinations.shape[1]/2)
combinations = combinations.tolist()
dim = len(G.nodes())
print("Done!")
print("Calculate embedding...")
import time
stime = time.time()
chi, heat_print, taus = graphwave_alg(G, np.linspace(0,100,25), taus='auto', verbose=True)
import pdb;pdb.set_trace()
print("Done! Take {}s", time.time()-stime)
combinations_list =  [(set(combination[:linearSize]), set(combination[linearSize:])) for combination in combinations]
non_combinations = []
while (len(non_combinations) < len(combinations)):
    non_combination = np.random.choice(np.arange(dim), size=(2*linearSize), replace=False)
    if (set(non_combination[:linearSize]), set(non_combination[linearSize:])) in combinations_list \
        or (set(non_combination[linearSize:]), set(non_combination[:linearSize])) in combinations_list:
        continue

    non_combinations.append(non_combination.tolist())

sims_cos = []
sims_dis = []
labels = []

for combination in combinations:
    labels.append(1)
    sims_cos.append(sim_vec(chi, combination))
    sims_dis.append(sim_vec(chi, combination, cosin=False))
import pdb;pdb.set_trace()
for non_combination in non_combinations:
    labels.append(0)
    sims_cos.append(sim_vec(chi, non_combination))
    sims_dis.append(sim_vec(chi, non_combination, cosin=False))
pdb.set_trace()
print("cosin: ",roc_auc_score(labels, sims_cos))
print("dis: ",roc_auc_score(labels, sims_dis))
