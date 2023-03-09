import numpy as np
import pandas as pd
from ampligraph.latent_features import ComplEx
from ampligraph.utils import save_model

dataset = 'codex'
embedding = 'complex'

# load dataset
prefix = f"../../../datasets/{dataset}"
codex_train = pd.read_csv(f"{prefix}/train.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_valid = pd.read_csv(f"{prefix}/valid.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_test = pd.read_csv(f"{prefix}/test.txt", delim_whitespace=True, names=['s', 'p', 'o']).to_numpy()
codex_filter = np.concatenate([codex_train, codex_valid, codex_test], 0)

model = ComplEx(k=350,
                epochs=4000, eta=30,
                loss='multiclass_nll',
                regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 3},
                optimizer='adam', optimizer_params={'lr': 0.00005},
                seed=0, batches_count=100, verbose=True)

early_stopping_params = {'x_valid': codex_valid[::2],
                         'criteria': 'mrr',
                         'x_filter': codex_filter,
                         'stop_interval': 4,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(codex_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

save_model(model, f'../../../models/{dataset}/{dataset}_{embedding}.pkl')
