from .tokenizer import SimpleTokenizer
from .reader import ReaderFactory

import collections
import numpy as np
import warnings
import pdb

def get_vocab_size(corpus):
    """ words are {0, 1, ..., n_words - 1}"""
    vocabulary_size = 1
    for idx, center_word_id in enumerate(corpus):
        if center_word_id + 1> vocabulary_size:
            vocabulary_size = center_word_id + 1
    print("vocabulary_size={}".format(vocabulary_size))
    return  vocabulary_size

def build_cooccurance_dict(data, skip_window, vocabulary_size):
    cooccurance_count = collections.defaultdict(collections.Counter)
    for idx, center_word_id in enumerate(data):
        if center_word_id > vocabulary_size:
            vocabulary_size = center_word_id
        for i in range(max(idx - skip_window - 1, 0), min(idx + skip_window + 1, len(data))):
            cooccurance_count[center_word_id][data[i]] += 1
        cooccurance_count[center_word_id][center_word_id] -= 1
    return cooccurance_count

from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
def worker(pij_dok,pi,pj,k,pair):
    i,j=pair
    x = np.log(pij_dok) - np.log(pi*pj) - np.log(k)
    x = np.array(x)[0][0]
    if np.isinf(x) or np.isnan(x):
        x = 0
    if x>0:
        return [i,j,x]
    else:
        return []

def construct_coo_matrix(data, k, skip_window):
    vocabulary_size = get_vocab_size(data)
    cooccur = build_cooccurance_dict(data, skip_window, vocabulary_size)

    rows = []
    cols = []
    data = []
    print("Constructing Nij...")
    for i in tqdm(range(vocabulary_size)):
        for j in range(vocabulary_size):
            if cooccur[i][j] != 0:
                rows.append(i)
                cols.append(j)
                data.append(cooccur[i][j])
    Nij = coo_matrix((np.array(data), (np.array(rows), np.array(cols))),shape=(vocabulary_size,vocabulary_size))
    Ni = np.sum(Nij, axis=1)
    tot = np.sum(Nij)
    with warnings.catch_warnings():
        """log(0) is going to throw warnings, but we will deal with it."""
        warnings.filterwarnings("ignore")

        Pij = Nij / tot
        Pi = Ni / np.sum(Ni)
        # c.f.Neural Word Embedding as Implicit Matrix Factorization, Levy & Goldberg, 2014
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import as_completed
        import multiprocessing as mp
        #idx_pairs = []
        #for i in range(vocabulary_size):
        #    for j in range(vocabulary_size):
        #        idx_pairs.append((i,j))

        rows = []
        cols = []
        data = []
        results = []
        #Pij_dok = Pij.todok()
        Pij_csr = Pij.tocsr()
        print("Constructing Pij...")
        def worker_(pair):
            i,j=pair
            x = np.log(Pij_dok[i,j]) - np.log(Pi[i]*Pi[j]) - np.log(k)
            x = np.array(x)[0][0]
            if np.isinf(x) or np.isnan(x):
                return []
            if x>0:
                return [i,j,x]
            else:
                return []
        """
        import time
        stime = time.time()
        for i in tqdm(range(vocabulary_size)):
            #pool = mp.Pool(processes=4)
            #results = [pool.apply(worker, args=(Pij_dok[i,j], Pi[i], Pi[j], k,(i,j),)) for j in range(vocabulary_size)]
            #results = [i for i in results if len(i) >0]

            with ThreadPoolExecutor(max_workers = 8) as executor:
                future_workers = [executor.submit(worker, (i,j)) for j in range(vocabulary_size)]
                #results = executor.map(worker, idx_pairs)
                for future in tqdm(as_completed(future_workers)):
                    try:
                        result = future.result()
                    except:
                        pass
                    else:
                        if len(result)>0:
                            results.append(result)

        print("Running time: ", time.time()-stime)
        rows, cols, data = zip(*results)
        """
        PMI = csr_matrix((vocabulary_size, vocabulary_size))
        for i in tqdm(range(vocabulary_size)):
            tmp = np.log(Pij_csr.getrow(i)/(Pi[i]*Pi.T*k))
            tmp[np.isnan(tmp)] =0
            tmp[np.isinf(tmp)]=0
            tmp[tmp<0] =0
            tmp = coo_matrix(tmp)
            #import pdb;pdb.set_trace()
            data.append(tmp.data.astype(np.float16))
            cols.append(tmp.col.astype(np.int16))
            rows.append(np.array([i]*tmp.col.shape[0]).astype(np.int16))

            #import pdb;pdb.set_trace()
            continue
            for j in range(vocabulary_size):
                #import time
                #stime = time.time()
                #a= Pij_dok[i,j]
                #b = Pi[i]
                #c = Pi[j]
                #print("Reading time ", time.time()-stime)
                #stime = time.time()
                #x= np.log(a)-np.log(b*c) - np.log(k)
                #print("CPU time ", time.time()-stime)
                x = np.log(Pij_dok[i,j]) - np.log(Pi[i]*Pi[j]) - np.log(k)
                x = np.array(x)[0][0]
                if np.isinf(x) or np.isnan(x):
                    x = 0
                if x>00:
                    rows.append(i)
                    cols.append(j)
                    data.append(x)
        #"""
        data = np.concatenate(data)
        cols = np.concatenate(cols)
        rows = np.concatenate(rows)
        PMI = coo_matrix((data, (rows, cols)),shape=(vocabulary_size,vocabulary_size))
        #PMI = coo_matrix((np.array(data), (np.array(rows), np.array(cols))),shape=(vocabulary_size,vocabulary_size))
        #import time
        #stime = time.time()
        #PMI = coo_matrix(PMI)
        #print("Convert time ", time.time()-stime)
    return PMI, Nij


def construct_matrix(data, k, skip_window):
    vocabulary_size = get_vocab_size(data)
    cooccur = build_cooccurance_dict(data, skip_window, vocabulary_size)

    Nij = np.zeros([vocabulary_size, vocabulary_size])
    for i in range(vocabulary_size):
        for j in range(vocabulary_size):
            Nij[i,j] += cooccur[i][j]
    Ni = np.sum(Nij, axis=1)
    tot = np.sum(Nij)
    with warnings.catch_warnings():
        """log(0) is going to throw warnings, but we will deal with it."""
        warnings.filterwarnings("ignore")
        Pij = Nij / tot
        Pi = Ni / np.sum(Ni)
        # c.f.Neural Word Embedding as Implicit Matrix Factorization, Levy & Goldberg, 2014
        PMI = np.log(Pij) - np.log(np.outer(Pi, Pi)) - np.log(k)
        PMI[np.isinf(PMI)] = 0
        PMI[np.isnan(PMI)] = 0
    return PMI, Nij

def generate_matrix(afile, vocab_size, min_count, neg_samples, skip_window):
    reader = ReaderFactory.produce(afile[-3:])
    data = reader.read_data(afile)
    tokenizer = SimpleTokenizer()
    indexed_corpus = tokenizer.do_index_data(data,
            n_words=vocab_size,
            min_count=min_count)
    pmi, nij = construct_coo_matrix(indexed_corpus, neg_samples, skip_window)
    return pmi, tokenizer.dictionary, tokenizer.reversed_dictionary, nij

if __name__ == "__main__":
    vocabulary_size = 10000
    min_count = 100
    neg_samples = 1
    skip_window = 5
    pmi, dd, rd, nij = generate_matrix("data/word2vec/text8.zip", vocabulary_size, min_count, neg_samples, skip_window)
    pdb.set_trace()
