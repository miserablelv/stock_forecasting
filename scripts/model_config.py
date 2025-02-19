from hyperopt import hp
from hyperopt.pyll.base import scope

from hyperopt import Trials, fmin, tpe

import time
import torch

# comparison between improved_space_loss and space_loss (revisit)

data_split_ratios = {'train_ratio': 0.52, 'val_ratio': 0.12} # 60/40 or 52/48

general_treatment = {'leave_outliers': hp.choice('remove_outliers', [True, False]), # not implemented
                     # 'remove_seasonality': hp.choice('remove_seasonality', [True, False]),
                     'remove_seasonality': False,
                     # 'log': hp.choice('log', [True, False]),
                     'log': True,
                    # 'remove_trend': hp.choice('remove_trend', [True, False]),
                    'remove_trend': False,
                    'group_by_weeks': True}

normalization_dict = {'general_treatment': general_treatment,
                    'transformer': hp.choice('transformer', ['MinMaxScaler', 'StandardScaler', 'PowerTransformer']),
                    'treatment': hp.choice('treatment', ['static', 'hybrid', 'dynamic'])}
                     # 'trend_type': hp.choice('trend_type', ['additive', 'multiplicative']),
                     # 'trend_degree': scope.int(hp.uniform('trend_degree', 1, 3))}

model_specific_parameters = {'loss_function': hp.choice('loss_function', ['RMSELoss', 'L1Loss', 'SmoothL1Loss', 'HuberLoss']),
    'optimizer': hp.choice('optimizer', ['SGD', 'Adam', 'AdamW', 'NAdam']),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.003),
    'dropout': hp.uniform('dropout', 0.3, 0.8),
    'stateful': hp.choice('stateful', [False]),
    'weight_decay': hp.uniform('weight_decay', 1e-7, 1e-3),
    'gradient_clipping': scope.int(hp.uniform('gradient_clipping', 3, 8)),
    'momentum': hp.uniform('momentum', 0.82, 0.97),
    'num_layers': hp.choice('num_layers', [2, 3]),
    'hidden_units': scope.int(hp.uniform('hidden_units', 50, 400)),
    'model': hp.choice('model', ['ARNN']),
    'multi_or_univariate': 'multivariate'}

batching_params = {
    # 'batch_size': scope.int(hp.uniform('batch_size', 2, 6)),
    'batch_size': 1,
    'seq_length': scope.int(hp.uniform('seq_length', 30, 150)),
    'n_days': hp.choice('n_days', [2, 3, 4, 5, 6, 7]) # predicted steps or stride}
}

data_params = {'context_factor': 3,
               'train_on': hp.choice('train_on', ['current_ticker']),#, 'all_tickers']),
               'group_by_weeks': True,
               'normalize_all': False,
              }

training_params = {
    'num_epochs': 1,
    'step': 1,
    'training': hp.choice('training', ['alternate']),
}

improved_space_loss = {
    'data_split_ratios': data_split_ratios,
    'normalization': normalization_dict,
    'model_params': model_specific_parameters,
    'batching': batching_params,
    'data': data_params,
    'training': training_params,    
}

# convergencewarning
sarima_params = { 'p_param': hp.choice('p_param', [0, 1, 2]),
    'd_param': hp.choice('d_param', [0, 1]),
    'q_param': hp.choice('q_param', [0, 1, 2]),
    't_param': hp.choice('t_param', ['n','c','t','ct']),
    'P_param': hp.choice('P_param', [0, 1, 2]),
    'D_param': hp.choice('D_param', [0, 1]),
    'Q_param': hp.choice('Q_param', [0, 1, 2]),
    'm_param': hp.choice('m_param', [4, 13]), # weeks
} # using a different context factor for SARIMA?

xgboost_params = {
    "n_estimators": hp.choice('n_estimators', [50, 100, 200, 300, 500]),
    "max_depth": hp.choice('max_depth', [3, 5, 7, 10, 15]),
    "learning_rate_x": hp.choice('learning_rate_x', [0.001, 0.01, 0.05, 0.1, 0.2]),
    "objective": hp.choice('objective', ["reg:squarederror"]),
    "subsample": hp.choice('subsample', [0.5, 0.7, 0.8, 1.0]),
    "colsample_bytree": hp.choice('colsample_bytree',[0.5, 0.7, 0.8, 1.0]),
    "gamma": hp.choice('gamma', [0, 0.1, 0.2, 0.5]),
    "reg_alpha": hp.choice('reg_alpha', [0, 0.01, 0.1, 1]),
    "reg_lambda": hp.choice('reg_lambda', [1, 1.5, 2, 3]),
    "random_state": hp.choice('random_state', [42]),
    # "tree_method": ["hist"],  # Faster computation for large datasets
}

space_loss = {
    'train_ratio':0.52,
    'val_ratio':0.12,
    'normalization': normalization_dict,
    'loss_function': hp.choice('loss_function', ['RMSELoss']), #, 'L1Loss', 'SmoothL1Loss', 'HuberLoss']),
    'optimizer': hp.choice('optimizer', ['SGD', 'Adam', 'AdamW', 'NAdam']),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.001),
    'dropout': hp.uniform('dropout', 0.3, 0.8),
    # 'batch_size': scope.int(hp.uniform('batch_size', 2, 6)), # NEEDS FIX !! i think it's fixed?
    'batch_size': 4,
    # 'seq_length': scope.int(hp.uniform('seq_length', 30, 200)),
    'seq_length': 100,
    # 'n_days': hp.choice('n_days', [2, 3, 4, 5, 6, 7]), # predicted steps or stride
    'n_days': 4,
    'stateful': hp.choice('stateful', [False]), # NO USO, PODRÍA HACER COMPARACIÓN RÁPIDA
    'ticker': hp.choice('ticker', ['SPY']), # NO USO
    'trend_type': hp.choice('trend_type', ['additive', 'multiplicative']), # NO USO
    'trend_degree': scope.int(hp.uniform('trend_degree', 1, 3)), # NO USO
    'weight_decay': hp.uniform('weight_decay', 1e-7, 1e-3),
    'gradient_clipping': scope.int(hp.uniform('gradient_clipping', 2, 8)),
    'momentum': hp.uniform('momentum', 0.82, 0.97),
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    # 'context_factor': hp.choice('context_factor', [4, 5, 6]),
    'context_factor': 5,
    'variable_context_size': hp.choice('variable_context_size', [True, False]),
    'hidden_units': scope.int(hp.uniform('hidden_units', 50, 400)),
    'model': hp.choice('model', ['ARNN']),
    'prime_hidden': hp.choice('prime_hidden', ['True', 'False']),
    'use_dataloader': True,
    'normalize_all': hp.choice('normalize_all', ['True', 'False']),
    'reuse_hidden_state': hp.choice('reuse_hidden_state', ['True', 'False']),
    'lstm': 'multivariate',
    'train_on': hp.choice('train_on', ['current_ticker']),#, 'all_tickers'])
    'sarima': sarima_params,
    'xgboost': xgboost_params,
    'num_epochs': 10,
    # 'group_by_weeks': True,
    'training': hp.choice('training', ['alternate']),
    'models_to_select': 5,
    'num_features': 17, # find a better way to do it
    # 'step': hp.choice('step', [1, 2, 3, 4]),
    'step': 1,
    'num_filters': hp.choice('num_filters', [16, 32, 64, 128]),
    'kernel_size': hp.choice('kernel_size', [3, 5, 7, 9, 11, 15]) # they are correlated
}