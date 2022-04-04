import pandas        as pd 
import gc 
import pickle
import numpy.ma as ma
from sklearn.metrics import make_scorer
import lightgbm      as lgbm
import sklearn
from skopt import BayesSearchCV
import numpy         as np 
from typing import Tuple
from skopt.space import Real, Categorical, Integer
    

version = 4
n_folds = 3
seed    = 99

print('Loading data...')
train = pd.read_pickle('~/data/raw/train.pkl').reset_index()
target   = 'target'
features = [col for col in train if col.startswith('f_')]

# Define custom cross-validation splitter
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
    return np.corrcoef(y_true, y_pred)[0][1]



scorer = make_scorer(corr)

early_stopping = lgbm.early_stopping(stopping_rounds=10)

def corr_eval(y_true, y_pred):
    return ('correlation', np.corrcoef(y_true, y_pred)[0][1], True)

params = {
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metrics': 'rmse'}

search_spaces = dict(
    max_depth        = Integer(50, 250),
    num_leaves       = Integer(200, 400),
    n_estimators     = Integer(600, 1000),
    lambda_l1        = Real(7, 12),
    lambda_l2        = Real(0.01, 0.05),
    min_data_in_leaf = Integer(1000, 2500),
    max_bin          = Integer(100, 300),
    feature_fraction = Real(0.6, 0.9),
    bagging_fraction = Real(0.95, 1.00),
    learning_rate    = Real(5e-4, 8e-3)
    )

print('Running grid search...')
bs = BayesSearchCV( estimator     = lgbm.LGBMRegressor(**params), 
                    search_spaces = search_spaces,
                    scoring       = scorer,
                    n_jobs        = 2,
                    verbose       = 0,
                    n_iter        = 30,
                    cv            = GroupTimeSeriesSplit(n_folds=n_folds,
                                                         holdout_size=200, 
                                                         groups=train['time_id']))

    

bs.fit(train[features], train[target],  
       callbacks=[early_stopping],
       eval_set=[(train[features], train[target])], 
       eval_metric=corr_eval)

print('Saving results...')
pickle.dump(bs.best_estimator_,     open(f"model_lgbm_bayessearch_version{version}.pkl"    , "wb"))
pickle.dump(bs.cv_results_,         open(f"results_lgbm_bayessearch_version{version}.pkl"    , "wb"))
pickle.dump(bs.best_params_,        open(f"params_lgbm_bayessearch_version{version}.pkl"    , "wb"))


