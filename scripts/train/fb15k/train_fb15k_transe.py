import numpy as np

from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import TransE
from ampligraph.utils import save_model

dataset = 'fb15k'
embedding = 'transe'

fb = load_fb15k_237()
fb_train = fb["train"]
fb_valid = fb["valid"][::2]  # for early stopping
fb_test = fb["test"]
fb_filter = np.concatenate([fb_train, fb_valid, fb_test], 0)

model = TransE(k=400,
               epochs=4000, eta=30,
               loss='multiclass_nll',
               regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 2},
               optimizer='adam', optimizer_params={'lr': 0.0001},
               embedding_model_params={'norm': 1, 'normalize_ent_emb': False},
               seed=0, batches_count=64, verbose=True)

early_stopping_params = {'x_valid': fb_valid,
                         'criteria': 'mrr',
                         'x_filter': fb_filter,
                         'stop_interval': 4,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(fb_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

save_model(model, f'../../../models/{dataset}/{dataset}_{embedding}.pkl')
