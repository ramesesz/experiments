import tensorflow as tf
import numpy as np
import pandas as pd
from ampligraph.latent_features import ConvE
from ampligraph.utils import save_model

# add to the top of your code under import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def create_reciprocals(dataset):
    # INPUT Dataframe
    # OUTPUT numpy array of dataframe with reciprocals

    dataset_reci = dataset.copy()

    dataset_reci = dataset_reci.reindex(columns=["o", "p", "s"])  # swap s and o columns
    dataset_reci['p'] = dataset_reci['p'].astype(str) + '_reciprocal'  # adds reciprocal to relations

    return np.concatenate([dataset.to_numpy(), dataset_reci.to_numpy()], 0)


dataset = 'codex'
embedding = 'conve'

# load dataset
codex_train = pd.read_csv("../../../datasets/train.txt", delim_whitespace=True, names=['s', 'p', 'o'])
codex_valid = pd.read_csv("../../../datasets/valid.txt", delim_whitespace=True, names=['s', 'p', 'o'])
codex_test = pd.read_csv("../../../datasets/test.txt", delim_whitespace=True, names=['s', 'p', 'o'])

# add reciprocals
codex_train = create_reciprocals(codex_train)
codex_valid = create_reciprocals(codex_valid)
codex_test = create_reciprocals(codex_test)

codex_filter = np.concatenate([codex_train, codex_valid, codex_test], 0)

model = ConvE(k=300, epochs=500,
              loss='bce', loss_params={'label_smoothing': 0.1},
              optimizer='adam', optimizer_params={'lr': 0.0001},
              embedding_model_params={'conv_filters': 32,
                                      'conv_kernel_size': 3,
                                      'dropout_embed': 0.2,
                                      'dropout_conv': 0.1,
                                      'dropout_dense:': 0.3,
                                      'use_batchnorm': True,
                                      'use_bias': True},
              seed=0, batches_count=300, verbose=True)

model.fit(codex_train)

save_model(model, f'../../../models/{dataset}/{dataset}_{embedding}.pkl')
