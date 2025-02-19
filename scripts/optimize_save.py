from data import *
from trading_strategies import *
from optimization import *
from save import *
from visualize import visualize_predictions, visualize_models_losses


from hyperopt import hp
from hyperopt.pyll.base import scope

from hyperopt import Trials, fmin, tpe

import time

from collections import defaultdict

import os
from copy import copy

from torch.nn import MSELoss
from torch import sqrt, float32, tensor

import numpy as np
from scipy.stats import gmean, hmean
from statistics import median
from numpy import max

def get_model_func(model): # repeated
    if model=='LastKMedian':
        return get_naive_model
    elif model=='SARIMA':
        return get_statistical_model
    elif model == 'ARNN' or model == 'XGBoost':
        return get_ML_model
    else:
        return get_DL_model

import talib



def check_test_predictions(model_name, test_set):
    base_path = f"{os.getcwd()}/best_models/{model_name}"
    test_data = load_data(f"{base_path}/{test_set}_com_scale", scaled=False)
    test_targets = create_overlapping_targets(test_data, column='Open', window_size=4, step=1)
    test_predictions = load_predictions(f"{base_path}/{test_set}_com_scale").values.tolist()
    test_loss = calculate_loss(test_targets, test_predictions[:len(test_targets)], model_params)
    return

def retrain_model(model_name, test_set='val_2'):
    model, model_params, data = load_model(model_name, test_set)[0] # the model's weights might be non-defined or else loaded from the best instance, depending on whether the model was already trained

    model_params['variable_context_size'] = False

    trainval_data, test_data = get_trainval_data_split(data, model_params, test_set)

    scaled_trainval_data, scaled_test_data, detrender, deseasonalizer = apply_general_treatment(trainval_data, test_data, model_params['normalization']['general_treatment'])

    if model_instance_available(model_name, test_set) or model_name == "SARIMA":
        print(f"Already trained before {test_set}, loading model...")
    else:
        print(f"Model instance {model_name} not found")
        # model_params['num_epochs'] = 3
        if model_name != 'SARIMA' and model_name != 'Drift':
            print(f"Model is {model}")
            model_instance = train_model(model_name, model, model_params, scaled_trainval_data, validate_train=True, set=f"train_before_{test_set}")
            save_retrained_model(model_instance, model_name, loss=1, test_set=test_set) # NEED TO UPDATE THE LOSS
            
    if predictions_available(model_name, test_set):
        print(f"Predictions already prepared for set {test_set}")
    else:
        print(f"Predictions not yet prepared for set {test_set}")
        test_predictions, test_targets, scaled_test_predictions, scaled_test_targets, test_loss = model.validate_forward_dataloader(scaled_trainval_data, scaled_test_data)


        save_predictions_targets(model_name, test_set, test_predictions, test_targets, scale="log")
        save_predictions_targets(model_name, test_set, scaled_test_predictions, scaled_test_targets, scale="scaled")
    
    return model

from itertools import combinations

def generate_model_combinations(models):
    all_combinations = []
    n = len(models)
    for r in range(1, n + 1):
        all_combinations.extend(list(combinations(models, r)))
    return [list(combo) for combination in all_combinations]

# aggregation_functions = ['mean', 'gmean', 'hmean', 'median', 'max', 'blend1', 'blend2', 'blend3']
aggregation_functions = ['median', 'mean', 'max']

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
        print(f"Combination is {combination}")
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
    
            print(f"Min loss with {combination} and {aggregation_function}: {loss}")

    # save the best config
    best_predictions_df = pd.DataFrame(best_predictions, index=test_data.index)
    save_best_aggregations(best_combinations, best_aggregation_funcs, best_predictions_df, min_losses)
            
    return min_losses, best_combinations, best_aggregation_funcs

from save import load_best_aggregations

def optimize_strategies(strategy_params, indicators_params):
    model_combinations = load_best_aggregations()

    best_model_combination = model_combinations['combinations']
    print(f"Best models combination {best_model_combination}")
    best_model_aggregation = model_combinations['aggregations'] # [1] # take all that are not none

    best_indicators_strategy = optimize_indicators_strategy(strategy_params)
    best_model_strategy = optimize_model_strategies(best_model_combination, strategy_params, best_agg=best_model_aggregation, use="PREDICTIONS") 
    best_hybrid_strategy = optimize_model_strategies(best_model_combination, strategy_params, best_agg=best_model_aggregation, use="PREDICTIONS+INDICATORS") # different indicators than on pure indicators strategy

    save_strategies_ranking((best_indicators_strategy, best_model_strategy, best_hybrid_strategy))

    return best_indicators_strategy, best_model_strategy, best_hybrid_strategy


def prepare_data(params):
    data, params['num_features'] = read_data(params['group_by_weeks'])
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
            print(f"{model_name} best configuration already found")
            print(f"Do you want to remove it and optimize it again? Y/N")
            x = input()
            if x == 'Y' or x == 'y':
                remove_all_files(f"{path}{model_name}/")
                print("Training again...")
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

    print(f"\n\n\nBest profit using purely indicators {strategy_profit}%\n\n\n")
    print(f"Best sharpe {best_trial['result']['sharpe']}")

    target_path = f"{base_path}INDICATORS/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    save_strategy(base_path, strategy_params['use'], best_strategy_config, best_strategy, "val_3")

    return best_strategy
    

def optimize_model_strategies(model_combinations, strategy_params, best_agg="gmean", scale='og', use="PREDICTIONS+INDICATORS"):
    base_path = os.path.join(os.getcwd(), "best_strategies/")

    best_sharpe = -np.inf
    best_profit = -np.inf
    best_ovr_strategy_config = None
    best_scale, best_aggregation = None, None

    avg_log_profit = 0
    avg_scaled_profit = 0

    feed_path = os.path.join(os.getcwd(), "datasets/val_3_data.csv")
    
    for combination in model_combinations:
        for aggregation in [best_agg, "max"]:
            for scale in ["log", "scaled"]: # original
                predictions, targets = prepare_aggregation(combination, feed_path, agg_func=aggregation, set="val_3", scale=scale)
                
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
                    best_ovr_strategy_config['agg_func'] = aggregation
                    best_ovr_strategy_config['n_models'] = len(combination)
                    best_scale = scale
                    best_aggregation = aggregation
                if scale == "log":
                    avg_log_profit += strategy_profit
                else:
                    avg_scaled_profit += strategy_profit

    avg_log_profit /= (len(model_combinations) * 2)
    avg_scaled_profit /= (len(model_combinations) * 2)

    print(f"\n\n\nBest profit using {use}, aggregation {best_aggregation} and scale {best_scale} {best_profit}%, best sharpe {best_sharpe}\n\n\n")

    print(f"Avg log profit {avg_log_profit}, avg scaled profit {avg_scaled_profit}")
        
    save_strategy(base_path, use, best_ovr_strategy_config, best_ovr_strategy, "val_3")
    
    return best_ovr_strategy

from save import load_model
        

def optimize_predictions_aggregation(combinations):
    agg_params = {}
    objective = create_objective_aggregation(agg_params)
    trials = Trials()
    best = fmin(objective, combinations, algo=tpe.suggest, max_evals=3, trials=trials)
    best_trial = min((trial for trial in trials.trials),#) if trial['result']['strategy'].wins > 0), # avoid strategies that just buy and hold?
        key=lambda x: x['result']['loss']
    )
    return

aggregation_dict = {'mean': gmean,
                    'median': median,
                    'gmean': gmean,
                    'hmean': hmean,
                    'max': max
                   }

def get_predictions_df(predictions, reference_model, set): ## makes sense? they are already indexed
    test_data_path = f"{os.getcwd()}/datasets/{set}"
    test_data = load_data(test_data_path)
    predictions_df = pd.DataFrame(predictions, index=test_data.index)
    return predictions_df

from save import select_best_aggregation_number

def test_best_config(use, scale='log', taxes=True):
    # should actually load it from best_overall_config/
    strategy_path = f"{os.getcwd()}/best_strategies/{use}/"
    strategy_params = load_strategy_params(strategy_path)
    strategy_params["substract_taxes"] = taxes
    
    best_agg_conf = load_best_aggregation_params()
    best_n = select_best_aggregation_number()
    best_models = best_agg_conf['combinations'][2] # hardcoded '1'
    best_agg_func = best_agg_conf['aggregations'][2]
    # best_agg_func = 'hmean'
    
    feed_path = f"{os.getcwd()}/datasets/test_data.csv"

    predictions, targets = prepare_aggregation(best_models, feed_path, agg_func=best_agg_func, set="test", scale=scale)

    indicators = prepare_indicators(best_models[0], strategy_params['indicators'], set='test') # they can be prepared previously to

    strategy, profit = evaluate_strategy_config(None, strategy_params, predictions, targets, feed_path, indicators)
    
    sharpe_ratio = calculate_sharpe_ratio(strategy.results)

    strategy.results.to_csv(f"{os.getcwd()}/best_strategies/{use}/test_strategy_results.csv", index=True)
    
    return strategy, sharpe_ratio