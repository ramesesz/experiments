import sys
import os
import time
import pickle

from ampligraph.datasets import load_wn18rr
from ampligraph.utils import restore_model
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score
from experiments.utils.discovery import *


def display_aggregate_metrics(ranks):
    print('MR     :', mr_score(ranks))
    print('MRR    :', mrr_score(ranks))
    print('Hits@3 :', hits_at_n_score(ranks, 3))
    print('Hits@10:', hits_at_n_score(ranks, 10))


dataset = 'wn18'
embedding = 'transe'

# load datasets
wn = load_wn18rr()
wn_train = wn["train"]
wn_valid = wn["valid"]
wn_test = wn["test"]

# restore model
model = restore_model(f'../../../models/{dataset}/{dataset}_{embedding}.pkl')

with open(f'../../../logs/extraction/{dataset}_{embedding}.txt', 'w') as f:
    # Redirect standard output to the text file
    sys.stdout = f

    for strategy in ['random_uniform', 'entity_frequency', 'graph_degree', 'cluster_coefficient', 'cluster_triangles']:
        start = time.time()
        triples, ranks = discover_facts(wn_train,
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