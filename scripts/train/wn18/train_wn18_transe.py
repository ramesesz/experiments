import numpy as np
from ampligraph.datasets import load_wn18rr
from ampligraph.latent_features import TransE
from ampligraph.utils import save_model

dataset = 'wn18'
embedding = 'transe'

wn = load_wn18rr()
wn_train = wn["train"]
wn_valid = wn["valid"][::2]
wn_test = wn["test"]
wn_filter = np.concatenate([wn_train, wn_valid, wn_test], 0)

model = TransE(k=400,
               epochs=4000, eta=30,
               loss='multiclass_nll',
               regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 2},
               optimizer='adam', optimizer_params={'lr': 0.0001},
               embedding_model_params={'norm': 1, 'normalize_ent_emb': False},
               seed=0, batches_count=64, verbose=True)

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
