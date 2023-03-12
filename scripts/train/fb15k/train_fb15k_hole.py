import os
import numpy as np
from ampligraph.datasets import load_fb15k_237
from ampligraph.latent_features import *
from ampligraph.utils import save_model

dataset = 'fb15k'
embedding = 'hole'

# load dataset
fb = load_fb15k_237()
fb_train = fb["train"]
fb_valid = fb["valid"][::2]  # for early stopping
fb_test = fb["test"]
fb_filter = np.concatenate([fb_train, fb_valid, fb_test], 0)

model = HolE(k=350,
             epochs=4000, eta=50,
             loss='multiclass_nll',
             regularizer='LP', regularizer_params={'lambda': 0.0001, 'p': 2},
             optimizer='adam', optimizer_params={'lr': 0.0001},
             seed=0, batches_count=64, verbose=True)

early_stopping_params = {'x_valid': fb_valid[::2],
                         'criteria': 'mrr',
                         'x_filter': fb_filter,
                         'stop_interval': 4,
                         'burn_in': 0,
                         'check_interval': 50}

model.fit(fb_train,
          early_stopping=True,
          early_stopping_params=early_stopping_params
          )

directory_path = f'../../../models/{dataset}'

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory if it does not exist
    os.makedirs(directory_path)

save_model(model, f'{directory_path}/{dataset}_{embedding}.pkl')
