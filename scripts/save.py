import torch
import time
from time import sleep

from models.DL_models import get_DL_model
from models.ML_models import get_ML_model
from models.statistical_models import get_statistical_model
# from models.naive_models import get_naive_model

import numpy as np
import pandas as pd

import os

import json

import glob

from data import undo_general_treatment

import shutil

import re



def save_deploy_trained_model(trained_model):
    model_path = os.path.join(os.getcwd(), "best_overall_config/trained_deploy_model.pth")
    torch.save(trained_model.state_dict(), model_path)  # Save model state dictionary
    return


def save_datasets(data):
    train_data, val_1_data, val_2_data, val_3_data, test_data = data
    base_path = os.path.join(os.getcwd(), "datasets/")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    train_data.to_csv(base_path+"train_data.csv", index=True)
    val_1_data.to_csv(base_path+"val_1_data.csv", index=True)
    val_2_data.to_csv(base_path+"val_2_data.csv", index=True)
    val_3_data.to_csv(base_path+"val_3_data.csv", index=True)
    test_data.to_csv(base_path+"test_data.csv", index=True)
    return
    



# def save_best_strategy_for_model(model, index):
#     base_dir = os.path.join(os.getcwd(), f"best_models/{model}/")
#     files = os.listdir(base_dir)
    

#     # Get the list of all files in the current working directory
#     target_dir = os.path.join(base_dir, "best_model_config/")
#     os.makedirs(target_dir, exist_ok=True)
    
#     for file in files:
#         filename, ext = os.path.splitext(file)
        
#         if filename.endswith(f"_{index}"):
#             new_file_name = re.sub(r"_(\d+)(\.\w+)$", r"\2", file)
#             source_path = os.path.join(base_dir, file)
#             dest_path = os.path.join(target_dir, new_file_name)
            
#             shutil.copy(source_path, dest_path)
#             print(f"Copied {file} into {dest_path}")

#     print(f"All files have been copied to {target_dir}.")
#     return



# def save_best_overall_config(model_name):
#     base_dir = os.path.join(os.getcwd(), f"best_models/{model_name}/best_model_config/")
#     files = os.listdir(base_dir)

#     target_dir = os.path.join(os.getcwd(), f"best_overall_config/")
#     os.makedirs(target_dir, exist_ok=True)

#     for file in files:
#         new_file_name = re.sub(r"_(\d+)(\.\w+)$", r"\2", file)  # Match "_123" and replace with just the extension
#         source_path = os.path.join(base_dir, file)
#         dest_path = os.path.join(target_dir, new_file_name)
#         shutil.copy(source_path, dest_path)
#         print(f"Copied: {file} -> {dest_path}")

#     print(f"All matching files have been copied to {target_dir}")

#     # load model params to train model
#     params_path = f"{target_dir}model_config.json"
#     with open(params_path, 'r') as f:
#         model_params = json.load(f)
    
#     trained_model = train_model_for_test(model_params, target_dir)
#     if model_name != 'ARNN' and model_name != 'XGBoost':
#         torch.save(trained_model.state_dict(), f"{target_dir}trained_test_model.pth")
#     else:
#         torch.save(trained_model.model.state_dict(), f"{target_dir}trained_test_model.pth")

#     # could even save the test predictions too
#     return


def load_best_overall_config():
    base_dir = os.path.join(os.getcwd(), "best_overall_config/")
    
    data = load_idx_datasets(base_dir)
    params_path = f"{base_dir}model_config.json"
    with open(params_path, 'r') as f:
            model_params = json.load(f)

    strategy_params_df = pd.read_csv(f"{base_dir}strategy_config.csv")
    strategy_params = strategy_params_df.to_dict(orient="records")[0]

    return data, model_params, strategy_params
    

# def load_best_config_for_test():
#     path = os.path.join(os.getcwd(), "best_overall_config/")

#     # load data sets
#     data = load_idx_datasets(path)

#     # load model params
#     model_files = glob.glob(f"{path}/model_config.json")
#     if model_files: 
#         model_params_file = model_files[0]
#     else:
#         raise FileNotFoundError()
#     with open(model_params_file, 'r') as f:
#         model_params = json.load(f)

#     # load strategy params
#     strategy_files = glob.glob(f"{path}/strategy_config.csv") # maybe save it as json?
#     if strategy_files: 
#         strategy_params_file = strategy_files[0]
#     else:
#         raise FileNotFoundError()
#     strategy_df = pd.read_csv(strategy_params_file)
#     strategy_params = strategy_df.to_dict(orient="records")[0]

#     # load trained_model
#     model_path = f"{path}trained_test_model.pth"
#     get_model = get_model_func(model_params['model']) # encapsulate more
#     model = get_model(model_params)
#     if model_params['model'] != 'ARNN' and model_name != 'XGBoost':
#         model.load_state_dict(torch.load(model_path, weights_only=True))
#     else:
#         model.model.load_state_dict(torch.load(model_path, weights_only=False))
#     config = (data, model_params, strategy_params, model)
    
#     return config

def load_strategy_params(path): #model_name, strategy_type
    with open(path+"params.json", "r") as f:
        strategy_params = json.load(f)
    return strategy_params

    
def save_strategy(path, use, strategy_config, strategy, set):
    # save strategy config
    with open(f"{path}{use}/params.json", 'w') as f:
        json.dump(strategy_config, f)

    # save strategy results
    strategy.results.to_csv(f"{path}{use}/{set}_strategy_results.csv", index=True)    
        
    # save strategy instance
    
    return

def save_strategies_ranking(strategies):
    strategies_names = ["indicators", "predictions", "hybrid"]
    path = os.path.join(os.getcwd(), "best_strategies/")
    with open(f"{path}all_profit_pct.txt", "w") as f:
        for name, strategy in zip(strategies_names, strategies):
            f.write(f"Best profit achieved with {name}: {((strategy.results['Equity'].iloc[-1]/strategy.results['Equity'].iloc[0])-1)*100}\n")


# def load_best_strategy_per_model(model_names): # the best strategy for each model
#     base_path = os.path.join(os.getcwd(), "best_configs/")

#     strategies = []

#     for model_name in model_names:
#         path = os.path.join(os.getcwd(), f"best_models/{model_name}/best_config")
#         file = f"strategy_config_{i}.csv"
            
#         # Load strategy dictionary from CSV
#         strategy_path = f"{path}/strategy_config"
#         strategy_df = pd.read_csv(strategy_path)
#         strategy_config = strategy_df.to_dict(orient="records")[0]  # Convert first row to dict

#         strategies.append(strategy_config)
#         i += 1
            
#     return strategies

# def load_strategies_for_model(model_name): # the best strategy for each version of each model. ONLY FOR ENSEMBLE, AS THERE ARE NOT > 1 VERSIONS PER MODEL
#     strategies = []
#     i = 0
#     files_left = True

#     base_path = os.path.join(os.getcwd(), f"best_models/{model_name}")
    
#     while files_left:
#         strategy_path = f"{base_path}/strategy_config_{i}.csv"
#         if file_exists(strategy_path) == False:
#             print(f"{strategy_path} does not exist !")
#             files_left = False
#             continue

#         # Load strategy dictionary from CSV
#         strategy_df = pd.read_csv(strategy_path)
#         strategy_config = strategy_df.to_dict(orient="records")[0] # Convert first row to dict (needed?) 
#         strategy_config['idx'] = i

#         # Load strategy instance

#         # Load model number?
#         strategies.append(strategy_config)
#         i += 1

#     # print(f"strategies loaded {strategies}")
            
#     return strategies

# def load_best_model_strategy(model_name):
#     strategy_files = glob.glob(os.path.join(os.getcwd(), f"best_models/{model_name}/best_model_config/strategy_config_*.csv"))

#     if strategy_files:
#         strategy_path = strategy_files[0]
#     else:
#         raise FileNotFoundError(f"Best strategy for model {model_name} is not defined yet")
    
#     strategy_df = pd.read_csv(strategy_path)
#     strategy_config = strategy_df.to_dict(orient="records")[0]
#     print("\n\n LOADED \n\n")
#     return strategy_config

def save_models_ranking(min_losses):
    file_path = os.path.join(os.getcwd(), "best_models/best_models_ranked.txt")

    sorted_models = sorted(min_losses.items(), key=lambda x: x[1])
    
    with open(file_path, 'w') as f:
        for model_name, loss in sorted_models:
            f.write(f"{model_name}: {loss}\n")

    return


def load_models_ranking():
    file_path = os.path.join(os.getcwd(), "best_models/best_models_ranked.txt")
    with open(file_path, 'r') as f:
        model_names = [line.split(':')[0].strip() for line in f]
    return model_names


def save_indicators(path, indicators):
    rsi, macd, macd_signal, obv, atr = indicators
    print(f"RSI {rsi.head()}")#, RSI dataframe {pd.DataFrame(rsi).head()}")
    rsi.to_csv(f"{path}_rsi.csv", index=True)
    macd.to_csv(f"{path}_macd.csv", index=True)
    macd_signal.to_csv(f"{path}_macd_signal.csv", index=True)
    obv.to_csv(f"{path}_obv.csv", index=True)
    atr.to_csv(f"{path}_atr.csv", index=True)
    return

def save_data_set(data, path, idx): # too much granularity
    data.to_csv(path, index=idx)
    return

def save_agg_predictions(models, predictions, index=None):
    predictions_path = os.path.join(os.getcwd(), f"agg_predictions/{"_".join(models)}/{set}_predictions.csv")
    predictions_df = pd.DataFrame(predictions, index=index)
    predictions_df.to_csv(predictions_path, index=True)
    return

def save_best_aggregations(best_combinations, best_aggregation_funcs, best_predictions, best_losses):
    print(f"Saving best aggregations...")
    base_path = os.path.join(os.getcwd(), f"best_aggregations")

    text_path = f"{base_path}/performance_ranking.txt"

    with open(text_path, "w") as f:
        f.write("")

    for i in range(len(best_combinations)):
        target_path = f"{base_path}/{i+1}"

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        params_path = f"{base_path}/params.json"
        best_aggs_dict = {'combinations': best_combinations,
                          'aggregations': best_aggregation_funcs}
        with open(params_path, "w") as f:
            json.dump(best_aggs_dict, f)
    
        # make df
        
        best_predictions.to_csv(f"{target_path}/val_2_predictions.csv")

        with open(text_path, "a") as f:
            f.write(f"Performance with {i+1} models: {round(best_losses[i], 2)}\n")
    
    return

def load_best_aggregation_params():
    path = os.path.join(os.getcwd(), "best_aggregations/params.json")
    with open(path, "r") as f:
        model_params = json.load(f)
    return model_params


def select_best_aggregation_number():
    file_path = f"{os.getcwd()}/best_aggregations/performance_ranking.txt"

    performances = {}

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                n_models = int(parts[0].split()[2])
                loss = float(parts[1])
                performances[n_models] = loss

    min_loss = np.min(list(performances.values()))

    print(f"Min loss {min_loss}")
    print(f"Performances values {performances.values()}")

    best_combinations = [n_models for n_models, loss in performances.items() if performances[n_models] == min_loss]
    print(best_combinations)

    best_n = np.max(best_combinations)

    print(f"Best choice: {best_n} models with a loss of {min_loss}")
    
    return best_n
    
                
                


def save_predictions(path, predictions, index):
    predictions_df = predictions.to_csv()
    predictions = pd.read_csv(f"{path}_predictions.csv", index=True)
    return predictions

def load_predictions(path):
    predictions = pd.read_csv(f"{path}_predictions.csv", index_col=0, parse_dates=True)
    return predictions

def load_data(path, scaled=False):
    add = ""
    if scaled == True:
        add = "scaled_"
    data = pd.read_csv(f"{path}{add}_data.csv", index_col=0, parse_dates=True)
    return data

def load_indicators(path):
    rsi = pd.read_csv(f"{path}_rsi.csv", index_col=0, parse_dates=True)
    macd = pd.read_csv(f"{path}_macd.csv", index_col=0, parse_dates=True)
    macd_signal = pd.read_csv(f"{path}_macd_signal.csv", index_col=0, parse_dates=True)
    obv = pd.read_csv(f"{path}_obv.csv", index_col=0, parse_dates=True)
    atr = pd.read_csv(f"{path}_atr.csv", index_col=0, parse_dates=True)
    return (rsi, macd, macd_signal, obv, atr)


def load_datasets():
    base_path = os.path.join(os.getcwd(), "datasets/")
    
    train_data = pd.read_csv(f"{base_path}train_data.csv", index_col=0, parse_dates=True)
    val_1_data = pd.read_csv(f"{base_path}/val_1_data.csv", index_col=0, parse_dates=True)
    val_2_data = pd.read_csv(f"{base_path}/val_2_data.csv", index_col=0, parse_dates=True)
    val_3_data = pd.read_csv(f"{base_path}/val_3_data.csv", index_col=0, parse_dates=True)
    test_data = pd.read_csv(f"{base_path}/test_data.csv", index_col=0, parse_dates=True)
    return (train_data, val_1_data, val_2_data, val_3_data, test_data)


import shutil
    
def remove_all_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

def load_models_losses(models_list):
    losses_dict = {}
    base_path = os.path.join(os.getcwd(), "best_models")
    for model in models_list:
        losses_dict[model] = []
        file_path = f"{base_path}/{model}/validation_losses.txt"
        print(f"Trying to load models losses from {file_path}...")
        if (os.path.exists(file_path)):
            with open(file_path, "r") as file:
                for line in file:
                    loss = float(line.split(":")[1].strip())
                    # print(f"Loss {loss}")
                    losses_dict[model].append(loss)
            file_path = f"{base_path}/{model}/validation_losses.txt"
    return losses_dict

def save_model(model_name, model_config, model_instance, set='val_1'): # is it worth saving so many things?
    base_path = os.path.join(os.getcwd(), f"best_models/{model_name}")
    specific_path = f"{base_path}/{set}"
    if not os.path.exists(specific_path):
        os.makedirs(specific_path)

    params, (train_data, val_1_data, val_2_data, val_3_data, test_data), val_1_losses = model_config

    params_path = f"{base_path}/model_config.json"
    with open(params_path, 'w') as f:
        json.dump(params, f)

    file_path = f"{base_path}/validation_losses.txt"
    
    with open(file_path, "w") as file: # save the loss to the file
        for i, loss in enumerate(val_1_losses):
            file.write(f"Configuration {i+1} Validation Loss: {loss}\n")

    instance_path = f"{specific_path}/instance"#.pth"
    if model_name == 'ARNN':
        torch.save(model_instance.model.state_dict(), instance_path+".pth")
    elif model_name == 'SARIMA':
        # model_instance.model.save(instance_path+".pkl")
        print("No need to save SARIMA model")
    elif model_name == 'XGBoost':
        model_instance.model.save_model(instance_path+".json")
    else:
        torch.save(model_instance.state_dict(), instance_path+".pth")
        
        
    print("All configurations saved successfully.")
    return

    
def get_model_func(model):
    if model=='LastKMedian':
        return get_naive_model
    elif model=='SARIMA':
        return get_statistical_model
    elif model == 'ARNN' or model == 'XGBoost':
        return get_ML_model
    else:
        return get_DL_model


from statsmodels.tsa.statespace.sarimax import SARIMAXResults

def load_model(model_name, set='train'):
    configs = []

    base_path = os.path.join(os.getcwd(), f"best_models/{model_name}")
    
    params_path = f"{base_path}/model_config.json"
    
    with open(params_path, 'r') as f:
        model_params = json.load(f)

    get_model = get_model_func(model_params['model'])
    model = get_model(model_params)
    
    model_instance_path = glob.glob(f"{base_path}/{set}/instance.*")
    if model_instance_path:    
        if model_name == 'ARNN':
            model.model = model.model.load_state_dict(torch.load(model_instance_path[0], weights_only=False))
        elif model_name == 'XGBoost':
            model.model = model.model.load_model(model_instance_path[0])
        else:
            print(f"Loading {model_name} from {model_instance_path[0]}")
            model = torch.load(model_instance_path[0], weights_only=False) # why like this?
        
    data = load_datasets()
    
    # indicators = load_indicators(base_path) # only when preparing for the agent
    # predictions = load_idx_predictions(base_path, 'val_3') # only when preparing for the agent

    configs.append((model, model_params, data))
    
    print(f"{model_name} configurations loaded successfully.")

    return configs

def model_instance_available(model_name, test_set):
    model_route = glob.glob(f"{os.getcwd()}/best_models/{model_name}/{test_set}/instance.*")
    if model_route:
        return True
    return False

def predictions_available(model_name, test_set):
    base_path = f"{os.getcwd()}/best_models/{model_name}"
    if os.path.exists(f"{base_path}/{test_set}/log_predictions.csv"):
        return True
    return False

def load_dataset(set):
    return pd.read_csv(f"{os.getcwd()}/datasets/{set}_data.csv", index_col=0, parse_dates=True)

def save_predictions_targets(model_name, test_set, test_predictions, test_targets, scale):#scaled_test_data
    test_data = load_dataset(test_set)
    
    base_path = f"{os.getcwd()}/best_models/{model_name}/{test_set}/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    index = test_data.index
        
    pd.DataFrame(test_predictions[1:], index=index).to_csv(f"{base_path}/{scale}_predictions.csv")
    test_data = test_targets[:, 0].tolist()
    test_data.extend(test_targets[-1, 1:].tolist())
    pd.DataFrame(test_data, columns=['Open'], index=index).to_csv(f"{base_path}/{scale}_data.csv")

    # save transformers? 
    
    print("PREDICTIONS AND TARGETS SAVED SUCCESSFULLY")
    return


def save_retrained_model(model_instance, model_name, loss, test_set):
    base_path = os.path.join(os.getcwd(), f"best_models/{model_name}/{test_set}")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    model_path = f"{base_path}/instance"
    if model_name == 'ARNN':
        torch.save(model_instance.model, model_path+".pth") #.state_dict()
    elif model_name == 'XGBoost':
        model_instance.model.save_model(model_path+".json")
    else:
        torch.save(model_instance, model_path+".pth")  #.state_dict() 
 
    file_path = f"{base_path}/loss.txt"
    # Save the loss to the file
    with open(file_path, "w") as file:
        file.write(f"Best configuration retrained. {test_set} Loss: {loss}\n")

    print(f"--> Best configuration for {model_name} has been further trained and saved")
    
    return
