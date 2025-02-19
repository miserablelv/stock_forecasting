from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster

import torch
import pandas as pd

from data import *
from models.DL_models import *
from models.ML_models import *
from trading_strategies import *
from save import get_model_func

from visualize import visualize_predictions

import pyalgotrade
from pyalgotrade.barfeed import yahoofeed

from logging import log


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

import joblib

# optimize based on loss
def create_objective_loss(data):
    def objective(params):
        train_data, val_1_data, val_2_data, val_3_data, test_data = data
        scaled_train_data, scaled_val_1_data, detrender, deseasonalizer = apply_general_treatment(train_data, val_1_data, params['normalization']['general_treatment'])

        print(f"Params context len {params['context_len']}")

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

    print(f"\n\nPROFIT: {profit}%\n\n")    

    return strategy, profit

from save import load_model, load_strategy_params

def calculate_indicators_for_strategy(data, model_params, indicator_params, set='val_3'): # optimize hyperparameters man
    train_data, val_1_data, val_2_data, val_3_data, test_data = data
    if set == 'val_3':
        trainval_data = pd.concat((train_data, val_1_data, val_2_data, val_3_data))
        indicators_len = len(val_3_data)
        test_data = val_3_data
    else:
        trainval_data = pd.concat(data)
        indicators_len = len(test_data) # should be the same right? not always
    
    trainval_data = adjust_trainval_set(trainval_data, model_params)

    index = trainval_data.index

    adj_close = trainval_data['Adj Close'].values.astype(np.float64)
    high = trainval_data['High'].values.astype(np.float64)
    low = trainval_data['Low'].values.astype(np.float64)
    volume = trainval_data['Volume'].values.astype(np.float64)

    print(f"Adj close len {len(adj_close)}, index len {len(index)}")
    
    rsi = pd.DataFrame(talib.RSI(adj_close, timeperiod=indicator_params['rsi_time_period']), index=index)
    macd, macd_signal, _ = talib.MACD(
        volume,
        fastperiod=indicator_params['macd_fast_period'], slowperiod=indicator_params['macd_slow_period'], signalperiod=indicator_params['signal_period']
    )
    macd = pd.DataFrame(macd, index=index)
    macd_signal = pd.DataFrame(macd_signal, index=index)
    obv = pd.DataFrame(talib.OBV(adj_close, volume), index=index) # [-len(val_3_data):]
    atr = pd.DataFrame(talib.ATR(
        high,
        low,
        adj_close,
        timeperiod=indicator_params['atr_period']
    ), index=index)
    return rsi, macd, macd_signal, obv, atr

def prepare_indicators(model_name, indicator_params, set='val_3'):
    model_configs = load_model(model_name, set)
    model_instance, model_params, data = model_configs[0]
    indicators = calculate_indicators_for_strategy(data, model_params, indicator_params, set)
    return indicators

from hyperopt import pyll

from save import load_data

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

aggregation_dict = {'mean': gmean,
                    'median': median,
                    'gmean': gmean,
                    'hmean': hmean,
                    'max': max
                   }


def aggregate_predictions(predictions, configuration):
    print(f"\n\n\n\nAGGREGATING WITH {configuration}\n\n\n")
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


def get_aggregated_predictions(model_combination, agg_func, set="val_3", scale="log"):
    base_path = os.path.join(os.getcwd(), "best_models")
    all_predictions, all_test_data = [], []
    print(f"Model combination is {model_combination}")
    for model in model_combination:
        target_path = f"{base_path}/{model}/{set}/{scale}"
        all_predictions.append(load_predictions(target_path).values.tolist())
        all_test_data.append(load_data(target_path).values.tolist())
    max_predictions = np.max(all_predictions, axis=0)
    aggregated_predictions = aggregate_predictions(all_predictions, agg_func)
    aggregated_targets = aggregate_predictions(all_test_data, agg_func)
    return aggregated_predictions, aggregated_targets


def prepare_aggregation(models_combined, feed_path, agg_func, set, scale):  
    print(f"\n\n\n\nAGG FUNC in prepare_aggregaation {agg_func} \n\n\n\n")
    predictions, test_data = get_aggregated_predictions(models_combined, agg_func, set, scale)
    predictions = pd.DataFrame(predictions, index=pd.read_csv(feed_path, index_col=0, parse_dates=True).index)
    targets = pd.DataFrame(test_data, index=pd.read_csv(feed_path, index_col=0, parse_dates=True).index[:len(test_data)])
    
    return predictions, targets


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

from save import save_deploy_trained_model, load_best_overall_config, load_predictions, load_indicators

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