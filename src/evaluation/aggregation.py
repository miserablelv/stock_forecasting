import pandas as pd
import numpy as np
import hyperopt as hp
from scipy.stats import hmean, gmean

import os

from hyperopt import hp
from scipy.stats import gmean, hmean
from statistics import median

from persistence.persistence import load_predictions, load_data

# Define aggregation functions for DataFrames
def weighted_avg(df, weights):
    return (df * weights).sum() / weights.sum()

aggregation_functions = {
    'Average': lambda dfs: pd.concat(dfs).groupby(level=0).mean(),
    'Weighted Average': lambda dfs: weighted_avg(pd.concat(dfs).groupby(level=0).mean(), np.array([0.5, 0.5])),  # Example weights
    'Exponentially Weighted Average': lambda dfs: pd.concat(dfs).groupby(level=0).apply(
        lambda x: np.sum(x * np.exp(np.arange(len(x)) / len(x)) / np.sum(np.exp(np.arange(len(x)) / len(x))))
    ),
    'Median': lambda dfs: pd.concat(dfs).groupby(level=0).median(),
    'Max': lambda dfs: pd.concat(dfs).groupby(level=0).max(),
    'Min': lambda dfs: pd.concat(dfs).groupby(level=0).min(),
    'Geometric Mean': lambda dfs: pd.concat(dfs).groupby(level=0).apply(gmean),
    'Harmonic Mean': lambda dfs: pd.concat(dfs).groupby(level=0).apply(hmean),
}

# Update ensemble_params
ensemble_params = {
    'n_models': hp.choice('n_models', [1, 2, 3, 4, 5]),
    'agg_function': hp.choice('agg_function', list(aggregation_functions.values()))
}

aggregation_dict = {'mean': gmean,
                    'median': median,
                    'gmean': gmean,
                    'hmean': hmean,
                    'max': max
                   }


def aggregate_predictions(predictions, configuration):
    # print(f"\n\n\n\nAGGREGATING WITH {configuration}\n\n\n")
    if not configuration.startswith("blend"):
        # return aggregation_dict[configuration](predictions, axis=0).tolist()
        if configuration == 'median':
            return np.median(predictions, axis=0)
        elif configuration == 'mean':
            return np.mean(predictions, axis=0)
        elif configuration == 'max':
            return np.max(predictions, axis=0)
    else:
        gmean_predictions = np.array(gmean(predictions, axis=0))
        hmean_predictions = np.array(hmean(predictions, axis=0))
        median_predictions = np.median(predictions, axis=0)
        mean_predictions = np.mean(predictions, axis=0)
        
        if configuration == 'blend1':
            return (0.4 * median_predictions + 0.4 * gmean_predictions + 0.2 * mean_predictions).tolist()
        elif configuration == 'blend2':
            return (0.5 * median_predictions + 0.3 * gmean_predictions + 0.2 * hmean_predictions).tolist()
        elif configuration == 'blend3':
            return (0.4 * median_predictions + 0.3 * gmean_predictions + 0.2 * hmean_predictions + 0.1 * mean_predictions).tolist()
        else:
            raise Exception("Unknown configuration")


def get_aggregated_predictions(model_combination, agg_func, test_set="val_3", scale="log"):
    base_path = os.path.join(os.getcwd(), "best_models")
    all_predictions, all_test_data = [], []
    # print(f"Model combination is {model_combination}")
    for model in model_combination:
        target_path = f"{base_path}/{model}/{test_set}/{scale}"
        all_predictions.append(load_predictions(target_path).values.tolist())
        all_test_data.append(load_data(target_path).values.tolist())
    max_predictions = np.max(all_predictions, axis=0)
    aggregated_predictions = aggregate_predictions(all_predictions, agg_func)
    aggregated_targets = aggregate_predictions(all_test_data, agg_func)
    return aggregated_predictions, aggregated_targets


def prepare_aggregation(models_combined, feed_path, agg_func, test_set, scale):  
    # print(f"\n\n\n\nAGG FUNC in prepare_aggregaation {agg_func} \n\n\n\n")
    predictions, test_data = get_aggregated_predictions(models_combined, agg_func, test_set, scale)
    predictions = pd.DataFrame(predictions, index=pd.read_csv(feed_path, index_col=0, parse_dates=True).index)
    targets = pd.DataFrame(test_data, index=pd.read_csv(feed_path, index_col=0, parse_dates=True).index[:len(test_data)])
    
    return predictions, targets