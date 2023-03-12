import os
import numpy as np
from ampligraph.datasets import load_wn18rr
from ampligraph.latent_features import *
from ampligraph.utils import save_model

dataset = 'wn18'
embedding = 'hole'

# load dataset
wn = load_wn18rr()
wn_train = wn["train"]
wn_valid = wn["valid"][::2]
wn_test = wn["test"]
wn_filter = np.concatenate([wn_train, wn_valid, wn_test], 0)

model = HolE(k=200,
             epochs=4000, eta=20,
             loss='self_adversarial', loss_params={'margin': 1},
             optimizer='adam', optimizer_params={'lr': 0.0005},
             seed=0, batches_count=50, verbose=True)

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
