import numpy as np

from ampligraph.datasets import load_wn18rr
from ampligraph.latent_features import ComplEx
from ampligraph.utils import save_model

dataset = 'wn18'
embedding = 'complex'

wn = load_wn18rr()
wn_train = wn["train"]
wn_valid = wn["valid"][::2]  # for early stopping
wn_test = wn["test"]
wn_filter = np.concatenate([wn_train, wn_valid, wn_test], 0)

model = ComplEx(k=200,
                epochs=4000, eta=20,
                loss='multiclass_nll', loss_params={'margin': 1},
                regularizer='LP', regularizer_params={'lambda': 0.05, 'p': 3},
                optimizer='adam', optimizer_params={'lr': 0.0005},
                seed=0, batches_count=10, verbose=True)

early_stopping_params = {'x_valid': wn_valid,
                         'criteria': 'mrr',
                         'x_filter': wn_filter,
                         'stop_interval': 4,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(wn_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

save_model(model, f'../../../models/{dataset}/{dataset}_{embedding}.pkl')
