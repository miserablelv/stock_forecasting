# Forecasting & Time Series
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster

# Machine Learning & Deep Learning
import torch
from torch import sqrt, float32, tensor
from torch.nn import MSELoss
import numpy as np

# Data Handling
import pandas as pd
import os
import time
from collections import defaultdict
from copy import copy

# Custom Modules
from data.loader import *
from models.dl_models import *
from models.ml_models import *
from trading.strategies import *
from evaluation.visualize import visualize_predictions, visualize_models_losses
from persistence.persistence import *

# Logging
import logging

# Trading & Backtesting
import pyalgotrade
from pyalgotrade.barfeed import yahoofeed

# Hyperparameter Optimization
from hyperopt import hp, Trials, fmin, tpe
from hyperopt.pyll.base import scope

from models.dl_models import get_DL_model
from models.ml_models import get_ML_model
from models.statistical import get_statistical_model

# Metrics
from sklearn.metrics import root_mean_squared_error


criterion_dict = {
    'RMSELoss': torch.nn.MSELoss(),
    'L1Loss': torch.nn.L1Loss(),
    'SmoothL1Loss': torch.nn.SmoothL1Loss(),
    'HuberLoss': torch.nn.HuberLoss()
}

from hyperopt import STATUS_OK

def get_top_n_models(model_dict, n):
    sorted_models = sorted(model_dict.items(), key=lambda x: float(x[0]))

    top_n_models = [model for loss, model in sorted_models[:n]]
    
    return top_n_models


# optimize based on loss
def create_objective_loss(data):
    def objective(params):
    
        train_data, val_1_data, val_2_data, val_3_data, test_data = data
        scaled_train_data, scaled_val_1_data, detrender, deseasonalizer = apply_general_treatment(train_data, val_1_data, params['normalization']['general_treatment'])

        # print(f"Params context len {params['context_len']}")

        model_name = params['model']
        get_model = get_model_func(model_name)
        model = get_model(params)

        val_1_dataloader, original_length = get_dataloader(scaled_train_data, scaled_val_1_data, params, False, True)

        if model_name != 'SARIMA':
            model = train_model(model_name, model, params, scaled_train_data, validate_train=True)
            
        val_1_log_predictions, val_1_log_targets, val_1_scaled_predictions, val_1_scaled_targets, val_1_log_loss = model.validate_forward_dataloader(scaled_train_data, scaled_val_1_data)

        visualize_predictions(val_1_scaled_targets, val_1_scaled_predictions, val_1_log_loss, model_name, set='val_1')

        
        return {'loss': val_1_log_loss, 'status': STATUS_OK, 'params': params, 'data': (train_data, val_1_data, val_2_data, val_3_data, test_data), 'transformers': (detrender, deseasonalizer), 'instance': model, 'log_predictions': val_1_log_predictions, 'scaled_predictions': val_1_scaled_predictions, 'log_targets': val_1_log_targets, 'scaled_targets': val_1_scaled_targets}#, 'predictions': models_predictions}
    
    return objective

# from persistence.persistence import load_model, load_strategy_params

from hyperopt import pyll

from persistence.persistence import load_data

def create_objective_aggregation(nothing):
    def objective(agg_params):
        combination = agg_params[f'{agg_params['n']}']
        base_path = os.path.join(os.getcwd(), "best_models")
        preds = []
        print(f"Using combination {combination}")
        for model_name in combination:
            print(f"Model {model_name}")
            target_path = f"{base_path}/{model_name}/val_2"
            preds.append(load_predictions(target_path))
        train_path = f"{base_path}/{model_name}/val_1"
        val_1_data = load_data(train_path)
        val_2_data = load_data(target_path) # just once
        val_2_dataloader = get_dataloader(val_1_data, val_2_data, params) 
        print(f"Preds {preds}, data {data}")
            
        # load_predictions
        # perform_aggregation
        
        return {'loss': 0.1}
    return objective

from scipy.stats import gmean, hmean
from statistics import median
from numpy import max


# optimize based on profit
def create_objective_profit(feed_path, feed_model="SARIMA", predictions=None, targets=None):
    def objective(strategy_config):
        indicators = prepare_indicators(feed_model, strategy_config['indicators']) # optimize them. but not always need indicators?
    
        strategy, profit = evaluate_strategy_config(None, strategy_config, predictions, targets, feed_path, indicators)
        initial_equity = strategy.results['Equity'].iloc[0]
        final_equity = strategy.results['Equity'].iloc[-1]

        sharpe_ratio = calculate_sharpe_ratio(strategy.results)
        
        return {'loss': 1/(sharpe_ratio**2), 'status': STATUS_OK, 'strategy_config': strategy_config, 'strategy': strategy, 'sharpe': sharpe_ratio}

    return objective



def retrain_model(model_name, test_set='val_2'):
    model, model_params, data = load_model(model_name, test_set)[0] # the model's weights might be non-defined or else loaded from the best instance, depending on whether the model was already trained

    model_params['variable_context_size'] = False

    trainval_data, test_data = get_trainval_data_split(data, model_params, test_set)

    scaled_trainval_data, scaled_test_data, detrender, deseasonalizer = apply_general_treatment(trainval_data, test_data, model_params['normalization']['general_treatment'])

    if model_instance_available(model_name, test_set) or model_name == "SARIMA":
        print(f"📊 {model_name} already trained before {test_set}, loading model...\n")
    else:
        print(f"🚧 Model instance {model_name} trained up before set {test_set} not found\n")
        # model_params['num_epochs'] = 3
        if model_name != 'SARIMA' and model_name != 'Drift':
            # print(f"Model is {model}")
            model_instance = train_model(model_name, model, model_params, scaled_trainval_data, validate_train=True, set=f"train_before_{test_set}")
            save_retrained_model(model_instance, model_name, loss=1, test_set=test_set) # NEED TO UPDATE THE LOSS
            
    if predictions_available(model_name, test_set):
        print(f"✅ Predictions already prepared for {model_name} for set {test_set} 🎯\n")
    else:
        print(f"🚧 Predictions not yet prepared for {model_name} for set {test_set}\n")
        test_predictions, test_targets, scaled_test_predictions, scaled_test_targets, test_loss = model.validate_forward_dataloader(scaled_trainval_data, scaled_test_data)


        save_predictions_targets(model_name, test_set, test_predictions, test_targets, scale="log")
        save_predictions_targets(model_name, test_set, scaled_test_predictions, scaled_test_targets, scale="scaled")
    
    return model

from persistence.persistence import save_deploy_trained_model, load_best_overall_config, load_predictions, load_indicators

def deploy_best_config(): # revisit
    data, model_params, strategy_params = load_best_overall_config()
    
    all_data = pd.concat(data)
    all_data = adjust_trainval_set(all_data, model_params)
    print(f"all date {all_data.head()}")
    set_context_len(all_data, model_params)
    # all_data_scaled, None = apply_general_normalization
    # train_dataloader = get_dataloader(None, all_data, model_params, True)
    
    get_model = get_model_func(model_params['model'])
    model = get_model(model_params)
    trained_model = train_model(model_params['model'], model, model_params, all_data)
    
    save_deploy_trained_model(trained_model)

    return trained_model



##### OPTIMIZE_SAVE ###############


# def get_model_func(model): # repeated
#     if model=='LastKMedian':
#         return get_naive_model
#     elif model=='SARIMA':
#         return get_statistical_model
#     elif model == 'ARNN' or model == 'XGBoost':
#         return get_ML_model
#     else:
#         return get_DL_model




def check_test_predictions(model_name, test_set):
    base_path = f"{os.getcwd()}/best_models/{model_name}"
    test_data = load_data(f"{base_path}/{test_set}_com_scale", scaled=False)
    test_targets = create_overlapping_targets(test_data, column='Open', window_size=4, step=1)
    test_predictions = load_predictions(f"{base_path}/{test_set}_com_scale").values.tolist()
    test_loss = calculate_loss(test_targets, test_predictions[:len(test_targets)], model_params)
    return

from itertools import combinations

def generate_model_combinations(models):
    all_combinations = []
    n = len(models)
    for r in range(1, n + 1):
        all_combinations.extend(list(combinations(models, r)))
    return [list(combination) for combination in all_combinations]


def get_model_func(model):
    if model=='LastKMedian':
        return get_naive_model
    elif model=='SARIMA':
        return get_statistical_model
    elif model == 'ARNN' or model == 'XGBoost':
        return get_ML_model
    else:
        return get_DL_model


# aggregation_functions = ['mean', 'gmean', 'hmean', 'median', 'max', 'blend1', 'blend2', 'blend3']
# aggregation_functions = ['median', 'mean', 'max']

from evaluation.aggregation import *

def compare_all_models_combinations(models_list, test_set="val_2"):
    combinations = generate_model_combinations(models_list)

    base_path = os.path.join(os.getcwd(), "best_models")
    target_path = f"{base_path}/{models_list[0]}/{test_set}" # better save it in a general folder
    test_data = load_dataset(test_set)    
    
    total_configs = np.array([0, 0, 0, 0, 0, 0])
    total_losses = np.array([0, 0, 0, 0, 0, 0])
    total_losses = np.zeros(shape=(len(models_list)))
    min_losses = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    best_combinations = [None, None, None, None, None, None]
    best_aggregation_funcs = [None, None, None, None, None, None]
    best_predictions = None

    criterion = torch.nn.MSELoss()
    
    for combination in combinations:
        # print(f"Combination is {combination}")
        all_predictions, all_targets = [], []
        for model_name in combination:
            target_path = f"{base_path}/{model_name}/{test_set}/scaled"
            all_predictions.append(load_predictions(target_path).values.tolist())#[:len(targets)].tolist())
            all_targets.append(create_overlapping_targets(load_data(target_path), "Open", 4, 1)) # mucho sentido? no sé

        for aggregation_function in ["max", "mean", "median"]:
            n_models = len(combination)
            if n_models > 1:
                aggregated_predictions = aggregate_predictions(all_predictions, aggregation_function)
                aggregated_targets = aggregate_predictions(all_targets, aggregation_function)
            else:
                aggregated_predictions = all_predictions # no dimensionality issue?
                aggregated_targets = all_targets
            
            
            predictions = tensor(aggregated_predictions, dtype=float32).squeeze()
            targets = tensor(aggregated_targets, dtype=float32).squeeze()
            loss = sqrt(criterion(predictions[:len(targets)], targets)).item() # RMSE

            total_losses[n_models-1] += loss
            total_configs[n_models-1] += 1
            if loss < min_losses[n_models-1]:
                min_losses[n_models-1] = loss
                best_combinations[n_models-1] = combination
                best_aggregation_funcs[n_models-1] = aggregation_function
                best_predictions = aggregated_predictions

    best_index = np.argmin(min_losses)
    print(f"🏆 Min loss with models {best_combinations[best_index]} and aggregating with{best_aggregation_funcs[best_index]}: {np.min(min_losses)}")

    # save the best config
    best_predictions_df = pd.DataFrame(best_predictions, index=test_data.index)
    save_best_aggregations(best_combinations, best_aggregation_funcs, best_predictions_df, min_losses)
            
    return min_losses, best_combinations, best_aggregation_funcs

def optimize_strategies(strategy_params, indicators_params):
    model_combinations = load_best_aggregation_params()

    best_model_combinations = model_combinations['combinations'] # what about the rest?
    # print(f"Best models combination {best_model_combination}")
    best_model_aggregations = model_combinations['aggregations'] # [1] # take all that are not none
    
    # best_indicators_strategy = optimize_indicators_strategy(strategy_params)
    best_model_strategy = optimize_model_strategies(best_model_combinations, strategy_params, best_model_aggregations, use="PREDICTIONS") 
    best_hybrid_strategy = optimize_model_strategies(best_model_combinations, strategy_params, best_model_aggregations, use="PREDICTIONS+INDICATORS") # different indicators than on pure indicators strategy

    save_strategies_ranking((best_model_strategy, best_hybrid_strategy))

    return best_model_strategy, best_hybrid_strategy


def prepare_data(params):
    data, params['num_features'] = read_data(True)
    train_data, val_1_data, val_2_data, val_3_data, test_data = split_data(data, params)            
    set_context_len(train_data, params)
    return (train_data, val_1_data, val_2_data, val_3_data, test_data)

def optimize_models_loss(space_loss, models_list, criterion_list, timeframe_list, mode="slow", overwrite=False): # overwrite not checked yet
    # updated_models_list = copy(models_list)
    
    if mode=="slow":
        n_configs_for_model = len(criterion_list) * len(timeframe_list)
    else:
        n_configs_for_model = 1

    path = os.path.join(os.getcwd(), "best_models/")
    if not os.path.exists(path):
        os.makedirs(path)

    # choose training mode
    for model_name in models_list: # for naive models no training
        space_loss['model'] = model_name # important
        trials = Trials()
        min_losses = {}

        if os.path.exists(f"{path}{model_name}/validation_losses.txt"):
            print(f"\b🎯 {model_name} best configuration already found\n")
            print(f"🗑️ Do you want to remove it and optimize it again? Y/N")
            x = input()
            if x == 'Y' or x == 'y':
                remove_all_files(f"{path}{model_name}/")
                print("🚧 Training again...")
            else:
                continue

        # space_loss['step'] = space_loss['n_days']
        space_loss['num_epochs'] = 1
        space_loss['num_layers'] = 1
        space_loss['hidden_units'] = 50
        space_loss['context_factor'] = 5
        space_loss['seq_length'] = 50

        data = prepare_data(space_loss)
        save_datasets(data)
            
        objective = create_objective_loss(data)
        best = fmin(objective, space_loss, algo=tpe.suggest, max_evals=1, trials=trials)
        losses = []
        for trial in trials.trials:
            losses.append(trial['result']['loss'])
        best_trial = min(trials.trials, key=lambda x: x['result']['loss'])

        visualize_predictions(best_trial['result']['log_targets'], best_trial['result']['log_predictions'], best_trial['result']['loss'], model_name, set="val_1")
        
        best_config = (best_trial['result']['params'], best_trial['result']['data'], losses)
        best_instance = best_trial['result']['instance']
        min_losses[model_name] = best_trial['result']['loss']
        # we will rank them based on best loss

        save_model(model_name, best_config, best_instance, "val_1")

        save_predictions_targets(model_name, "val_1", best_trial['result']['scaled_predictions'], best_trial['result']['scaled_targets'], scale="scaled")
        save_predictions_targets(model_name, "val_1", best_trial['result']['log_predictions'], best_trial['result']['log_targets'], scale="log")
    
    save_models_ranking(min_losses)
    
    models_losses = load_models_losses(models_list)
    visualize_models_losses(models_losses)

    return models_losses

    # joblib.dump(my_scaler, 'train_scaler.bin')
    # my_scaler = joblib.load('scaler.gz')

def evaluate_strategy_config(trained_model, strategy_params, test_predictions, test_data, feed_path, val_indicators):
    instrument = str.replace(feed_path, ".csv", "")
    feed = yahoofeed.Feed()
    feed.addBarsFromCSV(instrument, feed_path)

    if test_predictions is None:
        strategy = IndicatorsBasedStrategy(feed, instrument, test_data, strategy_params, val_indicators)
    else:
        strategy = PredictionBasedStrategy(feed, instrument, test_predictions, test_data, strategy_params, val_indicators)
    strategy.run()

    # profit = predictions_strategy.getProfit()
    profit = round(((strategy.getBroker().getEquity()-strategy.initial_investment)/strategy.initial_investment)*100, 2)

    # print(f"\n💰 PROFIT: {profit}%\n")    

    return strategy, profit

def optimize_indicators_strategy(strategy_params):
    strategy_params['use'] = "INDICATORS"
    base_path = os.path.join(os.getcwd(), "best_strategies/")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    feed_path = os.path.join(os.getcwd(), f"datasets/val_3_data.csv")
    targets = pd.read_csv(feed_path, index_col=0, parse_dates=True)
    
    objective = create_objective_profit(feed_path, targets=targets)
    trials = Trials()
    best = fmin(objective, strategy_params, algo=tpe.suggest, max_evals=20, trials=trials)
    best_trial = min((trial for trial in trials.trials),
        key=lambda x: x['result']['loss']
    )
    
    best_strategy_config = best_trial['result']['strategy_config']
    best_strategy = best_trial['result']['strategy']

    strategy_profit = ((best_strategy.results['Equity'].iloc[-1] / best_strategy.results['Equity'].iloc[0]) - 1) * 100

    print(f"📈 Best profit using purely indicators {strategy_profit}%\n")
    print(f"📊 Best sharpe {best_trial['result']['sharpe']}\n")

    target_path = f"{base_path}INDICATORS/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    save_strategy(base_path, strategy_params['use'], best_strategy_config, best_strategy, "val_3")

    return best_strategy
    


def optimize_model_strategies(model_combinations, strategy_params, model_aggregations, scale='og', use="PREDICTIONS+INDICATORS"):
    base_path = os.path.join(os.getcwd(), "best_strategies/")

    best_sharpe = -np.inf
    best_profit = -np.inf
    best_ovr_strategy_config = None
    best_scale, best_aggregation = None, None

    avg_log_profit = 0
    avg_scaled_profit = 0

    feed_path = os.path.join(os.getcwd(), "datasets/val_3_data.csv")

    i = 0
    for combination in model_combinations:
        for scale in ["log", "scaled"]: # original
            # print(f"Passing agg func {aggregation}")
            predictions, targets = prepare_aggregation(combination, feed_path, model_aggregations[i], test_set="val_3", scale=scale)
            
            target_path = f"{base_path}/{use}/"
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            strategy_params['use'] = use
        
            objective = create_objective_profit(feed_path, combination[0], predictions, targets)
            trials = Trials()
            best = fmin(objective, strategy_params, algo=tpe.suggest, max_evals=30, trials=trials)
            best_trial = min((trial for trial in trials.trials),
                key=lambda x: x['result']['loss']
            )
            best_strategy_config = best_trial['result']['strategy_config']
            best_strategy = best_trial['result']['strategy']
            # best_strategies_results.append(best_strategy.results)
        
            strategy_profit = ((best_strategy.results['Equity'].iloc[-1] / best_strategy.results['Equity'].iloc[0]) - 1) * 100
            strategy_sharpe = best_trial['result']['sharpe']
            if strategy_sharpe > best_sharpe:
                best_sharpe = strategy_sharpe
                best_profit = strategy_profit
                best_ovr_strategy = best_strategy
                best_ovr_strategy_config = best_strategy_config
                best_ovr_strategy_config['agg_func'] = model_aggregations[i]
                best_ovr_strategy_config['n_models'] = len(combination)
                best_ovr_strategy_config['scale'] = scale
            if scale == "log":
                avg_log_profit += strategy_profit
            else:
                avg_scaled_profit += strategy_profit

        i += 1

    avg_log_profit /= (len(model_combinations) * 2)
    avg_scaled_profit /= (len(model_combinations) * 2)

    print(f"\n\n\nBest profit using {use}, aggregation {best_ovr_strategy_config['agg_func']} and scale {best_ovr_strategy_config['scale']} is {best_profit}%, best sharpe {best_sharpe}\n\n\n")

    print(f"Avg log profit {avg_log_profit}, avg scaled profit {avg_scaled_profit}")
        
    save_strategy(base_path, use, best_ovr_strategy_config, best_ovr_strategy, "val_3")
    
    return best_ovr_strategy

from persistence.persistence import load_model
        

def optimize_predictions_aggregation(combinations): # using? should
    agg_params = {}
    objective = create_objective_aggregation(agg_params)
    trials = Trials()
    best = fmin(objective, combinations, algo=tpe.suggest, max_evals=3, trials=trials)
    best_trial = min((trial for trial in trials.trials),#) if trial['result']['strategy'].wins > 0), # avoid strategies that just buy and hold?
        key=lambda x: x['result']['loss']
    )
    return


from persistence.persistence import select_best_aggregation_number

from trading.metrics import calculate_sharpe_ratio

def test_best_config(use, scale='log', taxes=True):
    # should actually load it from best_overall_config/
    strategy_path = f"{os.getcwd()}/best_strategies/{use}/"
    strategy_params = load_strategy_params(strategy_path)
    strategy_params["substract_taxes"] = taxes
    strategy_params["use"] = use
    
    best_agg_conf = load_best_aggregation_params()
    best_n = select_best_aggregation_number()
    best_index = np.argmin(best_agg_conf['losses'])
    best_models = best_agg_conf['combinations'][best_index]
    best_agg_func = best_agg_conf['aggregations'][best_index]
    
    feed_path = f"{os.getcwd()}/datasets/test_data.csv"

    predictions, targets = prepare_aggregation(best_models, feed_path, best_agg_func, test_set="test", scale=scale)
    predictions_np = np.mean(predictions.to_numpy(), axis=1).flatten()
    targets_np = targets.to_numpy().flatten()

    benchmark_profit = ((targets_np[-1] - targets_np[0]) / targets_np[0]) * 100
    print(f"\nBuy and hold profit over test period: {benchmark_profit:.2f}%\n")

    rmse = root_mean_squared_error(targets_np, predictions_np)
    print(f"RMSE on test set: {rmse:.3f}\n")

    indicators = prepare_indicators(best_models[0], strategy_params['indicators'], test_set='test') # they can be prepared previously to

    if use == "INDICATORS":
        predictions = None
        
    strategy, profit = evaluate_strategy_config(None, strategy_params, predictions, targets, feed_path, indicators)
    
    print(f"Model + Strategy profit over test period:: {profit:.2f}%\n")
    alpha = profit - benchmark_profit
    print(f"Alpha generated: {alpha:.2f}%\n")
    sharpe_ratio = calculate_sharpe_ratio(strategy.results)
    print(f"Sharpe ratio (risk-adjusted returns): {sharpe_ratio:.2f}\n")
    actual_deltas = np.diff(targets_np)
    predicted_deltas = np.diff(predictions_np)
    directional_accuracy = np.mean(np.sign(actual_deltas) == np.sign(predicted_deltas)) * 100
    print(f"Directional accuracy (% of correct predictions): {directional_accuracy:.2f}%")

    strategy.results.to_csv(f"{os.getcwd()}/best_strategies/{use}/test_strategy_results.csv", index=True)
    
    return strategy, sharpe_ratio

"""
def test_best_config(use, scale='log', taxes=True):
    strategy_path = f"{os.getcwd()}/best_strategies/{use}/"
    strategy_params = load_strategy_params(strategy_path)
    strategy_params["substract_taxes"] = taxes
    strategy_params["use"] = use
    
    best_agg_conf = load_best_aggregation_params()
    best_index = np.argmin(best_agg_conf['losses'])
    best_models = best_agg_conf['combinations'][best_index]
    best_agg_func = best_agg_conf['aggregations'][best_index]
    
    feed_path = f"{os.getcwd()}/datasets/test_data.csv"

    # Get the raw arrays
    predictions, targets = prepare_aggregation(best_models, feed_path, best_agg_func, test_set="test", scale=scale)
    indicators = prepare_indicators(best_models[0], strategy_params['indicators'], test_set='test') 

    if use == "INDICATORS":
        predictions = None
        
    # 1. Evaluate the financial strategy
    strategy, profit = evaluate_strategy_config(None, strategy_params, predictions, targets, feed_path, indicators)
    sharpe_ratio = calculate_sharpe_ratio(strategy.results)

    # --- NEW FIX: Convert to 1D numpy arrays for safe mathematical indexing ---
    targets_arr = np.array(targets).flatten()
    
    # 2. Calculate Alpha (Excess Return vs Buy & Hold)
    benchmark_profit = ((targets_arr[-1] - targets_arr[0]) / targets_arr[0]) * 100
    alpha = profit - benchmark_profit

    print(f"\n📈 Benchmark Buy & Hold: {benchmark_profit:.2f}%")
    print(f"💰 Strategy Profit: {profit:.2f}%")
    print(f"🚀 Alpha Generated: {alpha:.2f}%")
    print(f"⚖️ Final Sharpe: {sharpe_ratio:.2f}\n")

    # 3. Evaluate the Machine Learning layer (if we have predictions)
    if predictions is not None:
        targets_arr = np.array(targets).flatten()
        predictions_arr = np.array(predictions)
        
        # --- NEW FIX: Handle Multi-Model / Ensemble Predictions ---
        if predictions_arr.ndim > 1:
            # If shape is (219, 4), average across the 4 models
            if predictions_arr.shape[0] == len(targets_arr):
                predictions_arr = np.mean(predictions_arr, axis=1)
            # If shape is (4, 219), average across the 4 models
            elif predictions_arr.shape[1] == len(targets_arr):
                predictions_arr = np.mean(predictions_arr, axis=0)
                
        predictions_arr = predictions_arr.flatten()
        
        # Calculate ML errors
        rmse = root_mean_squared_error(targets_arr, predictions_arr)
        
        # Calculate Naive Baseline (shifting targets by 1)
        naive_predictions = np.roll(targets_arr, shift=1)
        naive_predictions[0] = targets_arr[0] # Handle the edge case
        naive_rmse = root_mean_squared_error(targets_arr, naive_predictions)
        
        # Directional Accuracy
        actual_deltas = np.diff(targets_arr)
        predicted_deltas = np.diff(predictions_arr)
        directional_accuracy = np.mean(np.sign(actual_deltas) == np.sign(predicted_deltas)) * 100
        
        print(f"🧠 ML Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"🎯 Model RMSE: {rmse:.4f} | Naive RMSE: {naive_rmse:.4f}")
        if rmse < naive_rmse:
            print("✅ Model successfully beat the naive persistence baseline!")
        else:
            print("❌ Model failed to beat the naive persistence baseline.")

    strategy.results.to_csv(f"{os.getcwd()}/best_strategies/{use}/test_strategy_results.csv", index=True)
    
    return strategy
"""