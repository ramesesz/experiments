import os
import numpy as np
from ampligraph.datasets import load_wn18rr
from ampligraph.latent_features import *
from ampligraph.utils import save_model

dataset = 'wn18'
embedding = 'convkb'

# load dataset
wn = load_wn18rr()
wn_train = wn["train"]
wn_valid = wn["valid"][::2]
wn_test = wn["test"]
wn_filter = np.concatenate([wn_train, wn_valid, wn_test], 0)

model = ConvKB(k=200,
               epochs=500, eta=10,
               loss='multiclass_nll',
               loss_params={},
               optimizer='adam', optimizer_params={'lr': 0.0001},
               embedding_model_params={'num_filters': 32,
                                       'filter_sizes': 1,
                                       'dropout': 0.1},
               seed=0, batches_count=300, verbose=True)

early_stopping_params = {'x_valid': wn_valid[::2],
                         'criteria': 'mrr',
                         'x_filter': wn_filter,
                         'stop_interval': 4,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(wn_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

directory_path = f'../../../models/{dataset}'

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory if it does not exist
    os.makedirs(directory_path)

save_model(model, f'{directory_path}/{dataset}_{embedding}.pkl')
