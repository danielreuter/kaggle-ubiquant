# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:05:40 2022

@author: Daniel
"""


import pandas        as pd 
import gc 
import pickle
import numpy.ma as ma
from sklearn.metrics import make_scorer
from tensorflow.python.ops import math_ops
import sklearn
from sklearn.model_selection import GridSearchCV
import numpy         as np 
from typing import Tuple    
import tensorflow as tf
tf.random.set_seed(99)
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
tf.config.list_physical_devices('GPU') 

version = 5
n_folds = 3

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('Loading data...')
train = pd.read_pickle('~/data/raw/train.pkl').reset_index()
target   = 'target'
features = [col for col in train if col.startswith('f_')]

#"learning_rate": reciprocal(3e-4, 3e-2),


class GroupTimeSeriesSplit:
    """
    Custom class to create a Group Time Series Split. We ensure
    that the time id values that are in the testing data are not a part
    of the training data & the splits are temporal
    """
    def __init__(self, n_folds: int, holdout_size: int, groups) -> None:
        self.n_folds = n_folds
        self.holdout_size = holdout_size
        self.groups = groups

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_folds

    def split(self, X, y, groups) -> Tuple[np.array, np.array]:
        # Take the group column and get the unique values
        unique_time_ids = np.unique(self.groups.values)

        # Split the time ids into the length of the holdout size
        # and reverse so we work backwards in time. Also, makes
        # it easier to get the correct time_id values per
        # split
        array_split_time_ids = np.array_split(
            unique_time_ids, len(unique_time_ids) // self.holdout_size
        )[::-1]

        # Get the first n_folds values
        array_split_time_ids = array_split_time_ids[:self.n_folds]

        for time_ids in array_split_time_ids:
            # Get test index - time id values that are in the time_ids
            test_condition = self.groups.isin(time_ids)
            test_index = X.loc[test_condition].index.values

            # Get train index - The train index will be the time
            # id values right up until the minimum value in the test
            # data - we can also add a gap to this step by
            # time id < (min - gap)
            train_condition = self.groups < (np.min(time_ids))
            train_index = X.loc[train_condition].index.values

            yield train_index, test_index

def correlationMetric(x, y, axis=-2):
    """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
    x = tf.convert_to_tensor(x)
    y = math_ops.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n
    xvar = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
    yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xvar * yvar)
    return corr

# Define function to build an NN with flexible number of layers
def build_model(
                num_columns=300,
                hidden_units=[],
                dropout_rates=[0.1],
                lr=1e-3
                ):
    
    inp = tf.keras.layers.Input(shape = (num_columns, ))
    
    x = tf.keras.layers.BatchNormalization()(inp)    
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)    
        
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i])(x)
        
    out = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = tf.keras.models.Model(inputs = inp, outputs = out)
    model.compile(
                  optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                  loss = tf.keras.losses.MeanSquaredError(),
                  metrics = correlationMetric,
                  run_eagerly=True
                 )     
    
    return model
    


batch_size    = 2**15
epochs        = 50
hidden_units  = [512, 324, 256, 128]
dropout_rates = [0.3, 0.3, 0.2, 0.5, 0.3, 0.3] 
lr            = 1e-3    


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='correlationMetric',
                                                  patience=5,
                                                  mode='max',
                                                  min_delta=1e-3,
                                                  restore_best_weights=True)



print('Running grid search...')
model = build_model(hidden_units  = hidden_units,
                    dropout_rates = dropout_rates,
                    lr            = lr)



model.fit(train[features], train[target], 
          callbacks  = [early_stopping],
          batch_size = batch_size,
          epochs     = epochs)

print('Saving results...')
model.save(f'nn_model_version{version}')


