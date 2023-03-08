import tensorflow as tf
import numpy as np
from ampligraph.datasets import load_yago3_10
from ampligraph.latent_features import ConvE
from ampligraph.utils import save_model
from ampligraph.evaluation import evaluate_performance

# add to the top of your code under import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

dataset = 'yago'
embedding = 'conve'

yago = load_yago3_10(add_reciprocal_rels=True)
yago_train = yago["train"]
yago_valid = yago["valid"][::2]
yago_test = yago["test"]
yago_filter = np.concatenate([yago_train, yago_valid, yago_test], 0)

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
              seed=0, batches_count=500, verbose=True)

model.fit(yago_train)

save_model(model, f'../../../models/{dataset}/{dataset}_{embedding}.pkl')
