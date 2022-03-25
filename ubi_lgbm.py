import numpy         as np 
import pandas        as pd 
import gc 
import lightgbm      as lgbm
import pickle

from scipy.stats     import pearsonr
from collections     import defaultdict
from typing import Tuple


version = 3
seed    = 100


# Load data
train = pd.read_pickle('~/data/raw/train.pkl')
target   = 'target'
features = [col for col in train if col.startswith('f_')]

# Pick LGBM parameters
params = {'verbosity': -1,
  'n_jobs': -1,
  'seed': 100,
  'learning_rate': 0.022032103920961728,
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metric': 'rmse',
  'max_depth': 117,
  'num_leaves': 323,
  'n_estimators': 895,
  'lambda_l1': 9.05942864952896,
  'lambda_l2': 0.3245954750995574,
  'min_data_in_leaf': 1859,
  'max_bin': 233,
  'feature_fraction': 0.7136680164938948,
  'bagging_fraction': 0.983731275626164}



def rmse(y, x):
    temp = y-x
    temp = temp**2
    return temp.sum()

def run(params):    
    
    y = train['target']
    
    
    train_dataset = lgbm.Dataset(train.loc[:, features], y.loc[:])
    model = lgbm.train(
        params,
        train_set  = train_dataset, 
        verbose_eval=100
    )

    feature_importance = pd.DataFrame({
            'feature': features, 
            'importance': model.feature_importance()})    

    gc.collect()

  
    return model, feature_importance


# Make sets of features
base_features          = [col for col in train if col.startswith('f_')]
derived_features_all   = [col for col in train if col.startswith('all')]
derived_features_clust = [col for col in train if col.startswith('clust')]

features = base_features 

pickle.dump(run(params),     open(f"lgbm_results{version}.pkl"    , "wb"))
