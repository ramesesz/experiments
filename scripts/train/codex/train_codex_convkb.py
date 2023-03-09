import os
import numpy as np
import pandas as pd
from ampligraph.latent_features import *
from ampligraph.utils import save_model

dataset = 'codex'
embedding = 'convkb'

# load dataset
prefix = f"../../../datasets/{dataset}"
codex_train = pd.read_csv(f"{prefix}/train.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_valid = pd.read_csv(f"{prefix}/valid.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_test = pd.read_csv(f"{prefix}/test.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_filter = np.concatenate([codex_train, codex_valid, codex_test], 0)

model = ConvKB(k=200,
               epochs=500, eta=10,
               loss='multiclass_nll',
               loss_params={},
               # regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 3},
               optimizer='adam', optimizer_params={'lr': 0.0001},
               embedding_model_params={'num_filters': 32,
                                       'filter_sizes': 1,
                                       'dropout': 0.1},
               seed=0, batches_count=3000, verbose=True)

early_stopping_params = {'x_valid': codex_valid[::2],
                         'criteria': 'mrr',
                         'x_filter': codex_filter,
                         'stop_interval': 3,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(codex_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

directory_path = f'../../../models/{dataset}'

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory if it does not exist
    os.makedirs(directory_path)

save_model(model, f'{directory_path}/{dataset}_{embedding}.pkl')
