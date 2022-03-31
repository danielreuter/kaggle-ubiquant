import pandas        as pd 
import gc 
import pickle
import numpy.ma as ma
from sklearn.metrics import make_scorer
import sklearn
from sklearn.model_selection import BayesSearchCV
import numpy         as np 
from typing import Tuple 
import tensorflow as tf
tf.random.set_seed(99)
from keras.wrappers.scikit_learn import KerasRegressor
from skopt.space import Real, Categorical, Integer



version = 1
n_folds = 5
seed    = 100

print('Loading data...')
train = pd.read_pickle('~/data/raw/train.pkl')
target   = 'target'
features = [col for col in train if col.startswith('f_')]

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

def corr(y_true, y_pred):
    A = y_true.numpy()
    B = y_pred.numpy()
    
    return ma.corrcoef(ma.masked_invalid(A), ma.masked_invalid(B))[0][1]

# Define function to build an NN with flexible number of layers
def build_model(
                num_columns=300,
                hidden_unit1=256,
                hidden_unit2=256,
                hidden_unit3=256,
                hidden_unit4=256,
                hidden_unit5=256,
                dropout_rate1=0.3,
                dropout_rate2=0.3,
                dropout_rate3=0.3,
                dropout_rate4=0.3,
                dropout_rate5=0.3,
                lr=1e-3
                ):
    
    hidden_units  = [hidden_unit1, hidden_unit2, hidden_unit3, hidden_unit4, hidden_unit5]
    dropout_rates = [dropout_rate1, dropout_rate2, dropout_rate3, dropout_rate4, dropout_rate5]
    
    
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
                  metrics = sklearn.metrics.r2_score,
                  run_eagerly=True
                 )     
    
    return model
    

search_spaces = dict(
    batch_size    = [2**5],
    epochs        = [50],
    hidden_unit1  = Integer(200,1000),
    hidden_unit2  = Integer(200,1000),
    hidden_unit3  = Integer(200,1000),
    hidden_unit4  = Integer(200,1000),
    hidden_unit5  = Integer(200,1000),
    dropout_rate1 = Real(0.1, 0.5),
    dropout_rate2 = Real(0.1, 0.5),
    dropout_rate3 = Real(0.1, 0.5),
    dropout_rate4 = Real(0.1, 0.5),
    dropout_rate5 = Real(0.1, 0.5),
    lr            = Real(5e-4, 3e-3)
    )

scorer = make_scorer(sklearn.metrics.r2_score)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='r2_score',
                                                  patience=5,
                                                  mode='max',
                                                  min_delta=4e-4,
                                                  restore_best_weights=True)



print('Running grid search...')
bs = BayesSearchCV( estimator     = KerasRegressor(build_fn = build_model), 
                    fit_params    = {'callbacks': [early_stopping]},
                    search_spaces = search_spaces,
                    scoring       = scorer,
                    n_jobs        = -1,
                    verbose       = 1,
                    n_iter        = 100,
                    cv            = GroupTimeSeriesSplit(n_folds=n_folds,
                                                         holdout_size=200, 
                                                         groups=train['time_id']))




bs.fit(train[features], train[target])

print('Saving results...')
pickle.dump(bs,     open(f"nn_bayessearch_version{version}.pkl"    , "wb"))
