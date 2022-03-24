import pandas        as pd 
import gc 
import pickle


import torch
import torch.nn as nn

from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import TabularLearner

    
version = 1
seed    = 100
n_folds = 5


train = pd.read_pickle('~/data/raw/train.pkl')
target   = 'target'
features = [col for col in train if col.startswith('f_')]



def loss_func(y_predicted, y_target):
    return ((y_predicted - y_target)**2).mean()

def valid_func(y_predicted, y_target):
    return (((y_predicted - y_target)**2)[:, 0].mean())

class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        dropouts = [0.10, 0.10, 0.10, 0.10]
        hidden_size = [512, 256, 128]
        
        layers = [nn.BatchNorm1d(num_features)]
        in_size = num_features

        for i in range(len(hidden_size)):
            out_size = hidden_size[i]
            layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(nn.SiLU())  # SiLU aka swish
            in_size = out_size

        layers.append(nn.Dropout(dropouts[-1]))
        layers.append(nn.Linear(in_size, 1))
        
        self.model = torch.nn.Sequential(*layers)

    def forward(self, cat, cont):
        # fastai tabular passes categorical and continuous features separately
        x = self.model(cont)
        return x


BATCH_SIZE = 2**16
EPOCHS = 30

n_samples = len(train)
split_idx = int(n_samples*0.70)
val_idxs = range(split_idx, n_samples)

dls = TabularDataLoaders.from_df(
    train,
    cont_names=features,
    y_names='target',
    bs=BATCH_SIZE,
    valid_idx=val_idxs
)
    
dls.show_batch()

model = Model(
    num_features=len(features)
)

learner = TabularLearner(
    dls,
    model=model,
    loss_func=loss_func,
    metrics=[valid_func]
)

gc.collect()

learner.fit_one_cycle(EPOCHS)

pickle.dump(learner.model,     open(f"NNresults{version}.pkl"    , "wb"))
