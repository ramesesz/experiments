import tensorflow as tf
import numpy as np
from ampligraph.datasets import load_wn18rr
from ampligraph.latent_features import ConvE
from ampligraph.utils import save_model

# add to the top of your code under import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

dataset = 'wn18'
embedding = 'conve'

wn = load_wn18rr(add_reciprocal_rels=True)
wn_train = wn["train"]
wn_valid = wn["valid"][::2]
wn_test = wn["test"]
wn_filter = np.concatenate([wn_train, wn_valid, wn_test], 0)

model = ConvE(k=200, epochs=700,
              loss='bce', loss_params={'label_smoothing': 0.1},
              optimizer='adam', optimizer_params={'lr': 0.0001},
              embedding_model_params={'conv_filters': 32,
                                      'conv_kernel_size': 3,
                                      'dropout_embed': 0.2,
                                      'dropout_conv': 0.1,
                                      'dropout_dense:': 0.3,
                                      'use_batchnorm': True,
                                      'use_bias': True},
              seed=0, batches_count=100, verbose=True)

model.fit(wn_train)

save_model(model, f'../../../models/{dataset}/{dataset}_{embedding}.pkl')
