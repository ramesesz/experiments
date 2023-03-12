import os
import numpy as np
from ampligraph.datasets import load_yago3_10
from ampligraph.latent_features import *
from ampligraph.utils import save_model

dataset = 'yago'
embedding = 'convkb'

# load dataset
yago = load_yago3_10()
yago_train = yago["train"]
yago_valid = yago["valid"][::2]
yago_test = yago["test"]
yago_filter = np.concatenate([yago_train, yago_valid, yago_test], 0)

model = ConvKB(k=150,
               epochs=500, eta=10,
               loss='multiclass_nll',
               loss_params={},
               optimizer='adam', optimizer_params={'lr': 0.0001},
               embedding_model_params={'num_filters': 32,
                                       'filter_sizes': 1,
                                       'dropout': 0.1},
               seed=0, batches_count=3000, verbose=True)

early_stopping_params = {'x_valid': yago_valid[::2],
                         'criteria': 'mrr',
                         'x_filter': yago_filter,
                         'stop_interval': 4,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(yago_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

directory_path = f'../../../models/{dataset}'

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory if it does not exist
    os.makedirs(directory_path)

save_model(model, f'{directory_path}/{dataset}_{embedding}.pkl')
