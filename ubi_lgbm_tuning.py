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
n_folds = 5


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


# Load clusters
with open('feature_clusters.pickle', 'rb') as handle:
    clusters_dict = pickle.load(handle)
    del clusters_dict[-1]
    num_clusters = len(clusters_dict)
    
    
# Define feature engineering functions
def cut_distribution_broadly(df, columns, prefix):
 
    # Compute various new features from the cross-sectional distribution of features
    
    df[f'{prefix}_mean']   = df[columns].mean(axis=1)
    df[f'{prefix}_median'] = df[columns].median(axis=1)
    df[f'{prefix}_max']    = df[columns].max(axis=1)
    df[f'{prefix}_min']    = df[columns].min(axis=1)

    df[f'{prefix}_q25'] = df[columns].quantile(q=0.25, axis=1)
    df[f'{prefix}_q75'] = df[columns].quantile(q=0.75, axis=1)
    
    df[f'{prefix}_std']          = df[columns].std(axis=1)
    df[f'{prefix}_range']        = df[f'{prefix}_max'] - df[f'{prefix}_min']
    df[f'{prefix}_iqr']          = df[f'{prefix}_q75'] - df[f'{prefix}_q25']
    df[f'{prefix}_tails']        = df[f'{prefix}_range'] / df[f'{prefix}_iqr']
    df[f'{prefix}_dispersion']   = df[f'{prefix}_std'] / df[f'{prefix}_mean']
    df[f'{prefix}_dispersion_2'] = df[f'{prefix}_iqr'] / df[f'{prefix}_median']
    df[f'{prefix}_skew']         = df[columns].skew(axis=1)
    df[f'{prefix}_kurt']         = df[columns].kurt(axis=1)
    
    df[f'{prefix}_median-max'] = df[f'{prefix}_median'] - df[f'{prefix}_max']
    df[f'{prefix}_median-min'] = df[f'{prefix}_median'] - df[f'{prefix}_min']

def cut_distribution_finely(df, columns, prefix):
 
    df[f'{prefix}_q01'] = df[columns].quantile(q=0.01, axis=1)
    df[f'{prefix}_q05'] = df[columns].quantile(q=0.05, axis=1)
    df[f'{prefix}_q10'] = df[columns].quantile(q=0.10, axis=1)

    df[f'{prefix}_q90'] = df[columns].quantile(q=0.90, axis=1)
    df[f'{prefix}_q95'] = df[columns].quantile(q=0.95, axis=1)
    df[f'{prefix}_q99'] = df[columns].quantile(q=0.99, axis=1)
    

    df[f'{prefix}_q99-q95'] = df[f'{prefix}_q99'] - df[f'{prefix}_q95']
    df[f'{prefix}_q99-q90'] = df[f'{prefix}_q99'] - df[f'{prefix}_q90']
    df[f'{prefix}_q01-q05'] = df[f'{prefix}_q01'] - df[f'{prefix}_q05']
    df[f'{prefix}_q01-q10'] = df[f'{prefix}_q01'] - df[f'{prefix}_q10']

def cut_distribution(df, columns, prefix):
    cut_distribution_broadly(df, columns, prefix)
    cut_distribution_finely(df, columns, prefix)

def transform(df):
    
    features = [col for col in df if col.startswith('f_')]
    
    # Get features of the cross-sectional distribution using all 300 features    
    cut_distribution(df, features, 'all')
    
    # Now get features within-clusters (clustering algo: hierarchical)
    for k in range(num_clusters):
        cut_distribution_broadly(df, clusters_dict[k], f'clust{k}')
    
    # Now get features across-clusters
    df['clust_diff']  = df['clust0_mean'] - df['clust1_mean']
    df['clust_diff2'] = (df['clust0_mean'] - df['clust1_mean'])**2
    df['clust_diff3'] = df['clust0_std'] / df['clust1_std']
    df['clust_diff4'] = (df['clust0_std'] / df['clust1_std'])**2
    
# CV strategy -- code copied from https://www.kaggle.com/c/ubiquant-market-prediction/discussion/304036
class GroupTimeSeriesSplit:
    """
    Custom class to create a Group Time Series Split. We ensure
    that the time id values that are in the testing data are not a part
    of the training data & the splits are temporal
    """
    def __init__(self, n_folds: int, holdout_size: int, groups: str) -> None:
        self.n_folds = n_folds
        self.holdout_size = holdout_size
        self.groups = groups

    def split(self, X) -> Tuple[np.array, np.array]:
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
            test_condition = X['time_id'].isin(time_ids)
            test_index = X.loc[test_condition].index

            # Get train index - The train index will be the time
            # id values right up until the minimum value in the test
            # data - we can also add a gap to this step by
            # time id < (min - gap)
            train_condition = X['time_id'] < (np.min(time_ids))
            train_index = X.loc[train_condition].index

            yield train_index, test_index

# Define validation sets, derive new features
splitter = GroupTimeSeriesSplit(n_folds=n_folds, holdout_size=200, groups=train['time_id'])
transform(train)


def rmse(y, x):
    temp = y-x
    temp = temp**2
    return temp.sum()

def run(params):    
    
    y = train['target']
    train['preds'] = -1000
    scores = defaultdict(list)
    features_importance= []
    models = []
    
    def run_single_fold(fold, train_indices, valid_indices):
        train_dataset = lgbm.Dataset(train.loc[train_indices, features], y.loc[train_indices])
        valid_dataset = lgbm.Dataset(train.loc[valid_indices, features], y.loc[valid_indices])
        model = lgbm.train(
            params,
            train_set  = train_dataset, 
            valid_sets = [train_dataset, valid_dataset], 
            verbose_eval=100,
            early_stopping_rounds=50
        )
        preds = model.predict(train.loc[valid_indices, features])
        train.loc[valid_indices, "preds"] = preds

        
        scores[f'rmse_fold{fold+1}'] = rmse(    y.loc[valid_indices], preds)
        scores[f'corr_fold{fold+1}'] = pearsonr(y.loc[valid_indices], preds)[0]
        
        fold_importance_df = pd.DataFrame({
            'feature': features, 
            'importance': model.feature_importance(), 
            'fold': fold})
        
        features_importance.append(fold_importance_df)
        models.append(model)
        del train_dataset, valid_dataset
        gc.collect()

    for fold, (train_indices, valid_indices) in enumerate(splitter.split(train)):

        print('')
        print(f"=====================fold: {fold}=====================")
        print(f"train length: {len(train_indices)}, valid length: {len(valid_indices)}")
        run_single_fold(fold, train_indices, valid_indices)

                        
    return scores, models, pd.concat(features_importance, axis=0)


# Make sets of features
base_features          = [col for col in train if col.startswith('f_')]
derived_features_all   = [col for col in train if col.startswith('all')]
derived_features_clust = [col for col in train if col.startswith('clust')]

features = base_features 

pickle.dump(run(params),     open(f"results{version}.pkl"    , "wb"))
