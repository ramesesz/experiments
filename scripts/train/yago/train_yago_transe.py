import numpy as np
from ampligraph.datasets import load_yago3_10
from ampligraph.latent_features import TransE
from ampligraph.utils import save_model

dataset = 'yago'
embedding = 'transe'

yago = load_yago3_10()
yago_train = yago["train"]
yago_valid = yago["valid"][::2]
yago_test = yago["test"]
yago_filter = np.concatenate([yago_train, yago_valid, yago_test], 0)

model = TransE(k=350,
               epochs=4000, eta=30,
               loss='multiclass_nll',
               regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 2},
               optimizer='adam', optimizer_params={'lr': 0.0001},
               embedding_model_params={'norm': 1, 'normalize_ent_emb': False},
               seed=0, batches_count=100, verbose=True)

early_stopping_params = {'x_valid': yago_valid,
                         'criteria': 'mrr',
                         'x_filter': yago_filter,
                         'stop_interval': 4,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(yago_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

save_model(model, f'../../../models/{dataset}/{dataset}_{embedding}.pkl')
