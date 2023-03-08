import sys
import os
import time
import pickle

from ampligraph.utils import restore_model
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from experiments.utils.discovery import *

import pandas as pd


def display_aggregate_metrics(ranks):
    print('MR     :', mr_score(ranks))
    print('MRR    :', mrr_score(ranks))
    print('Hits@3 :', hits_at_n_score(ranks, 3))
    print('Hits@10:', hits_at_n_score(ranks, 10))


def create_reciprocals(dataset):
    # INPUT Dataframe
    # OUTPUT numpy array of dataframe with reciprocals

    dataset_reci = dataset.copy()

    dataset_reci = dataset_reci.reindex(columns=["o", "p", "s"])  # swap s and o columns
    dataset_reci['p'] = dataset_reci['p'].astype(str) + '_reciprocal'  # adds reciprocal to relations

    return np.concatenate([dataset.to_numpy(), dataset_reci.to_numpy()], 0)


dataset = 'codex'
embedding = 'conve'

# load datasets
codex_train = pd.read_csv("../../../datasets/codex/train.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_valid = pd.read_csv("../../../datasets/codex/valid.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_test = pd.read_csv("../../../datasets/codex/test.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()

# add reciprocals
codex_train = create_reciprocals(codex_train)
codex_valid = create_reciprocals(codex_valid)
codex_test = create_reciprocals(codex_test)

# restore model
model = restore_model(f'../../../models/{dataset}/{dataset}_{embedding}.pkl')

with open(f'../../../logs/extraction/{dataset}_{embedding}.txt', 'w') as f:
    # Redirect standard output to the text file
    sys.stdout = f

    for strategy in ['random_uniform', 'entity_frequency', 'graph_degree', 'cluster_coefficient', 'cluster_triangles']:
        start = time.time()
        triples, ranks = discover_facts(codex_train,
                                        model=model,
                                        top_n=500,
                                        max_candidates=500,
                                        strategy=strategy,
                                        seed=42)
        end = time.time()

        print('-------------------------------------')
        print('STRATEGY             : ', strategy)
        print('TOTAL TIME           : ', str(end - start))
        print('GENERATED CANDIDATES : ', len(triples))
        print('-------------------------------------')

        display_aggregate_metrics(ranks)

        results = {'triples': triples,
                   'ranks': ranks}

        # Set the directory path
        directory_path = f'../../../extract_results/{dataset}/{embedding}'

        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create the directory if it does not exist
            os.makedirs(directory_path)

        with open(f'{directory_path}/{strategy}.pkl', 'wb') as file:
            pickle.dump(results, file)

    # Restore standard output
    sys.stdout = sys.__stdout__