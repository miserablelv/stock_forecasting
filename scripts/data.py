import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.signal import detrend

import pandas as pd
from pandas_datareader import data as pdr

import yfinance as yf

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter

import talib

from sktime.transformations.series.detrend import Detrender, Deseasonalizer
from sktime.forecasting.trend import PolynomialTrendForecaster

from time import sleep

import sys

from joblib import dump, load


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

def read_ticker(ticker):
    data = None
    while data is None:
        try:
            data = yf.download(ticker)
        except:
            sleep(1)
    return data
        
import seaborn as sns
import missingno as msno

def read_data(group_by_weeks=False):
    spx = read_ticker('^SPX')
    vix = read_ticker('^VIX')
    dxy = read_ticker('DX-Y.NYB')
    
    aaii = pd.read_csv('sentiment.csv').set_index('Date') # make sure it is updated
    aaii.index = pd.to_datetime(aaii.index, format='%m-%d-%y')
    aaii = aaii.reindex(spx.index)
    
    spx['VIX'] = vix['Adj Close']
    spx['DXY'] = dxy['Adj Close']
    
    if group_by_weeks:
        for sentiment in ['Bullish', 'Bearish', 'Neutral']:
            spx[sentiment] = aaii[sentiment].str.replace('%', '').str.replace(',', '.').astype(float).interpolate(method='linear') / 100
        
        spx = spx.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Adj Close': 'last',
            'Volume': 'sum',
            'VIX': 'last',
            'DXY': 'last',
            'Bullish': 'max',
            'Neutral': 'max',
            'Bearish': 'max',
        })

        
    else:
        for sentiment in ['Bullish', 'Bearish', 'Neutral']:
            spx[sentiment] = aaii[sentiment].str.replace('%', '').str.replace(',', '.').astype(float).interpolate(method='linear') / 100

    # spx['Returns'] = spx['Adj Close'].pct_change()
    # spx['ROC'] = talib.ROC(spx['Adj Close'], timeperiod=10)

    # spx['Volatility_10'] = spx['Returns'].rolling(window=10).std()

    spx['VWAP'] = (spx['Adj Close'] * spx['Volume']).cumsum() / spx['Volume'].cumsum()
    
    spx['WMA'] = spx['Adj Close'].rolling(10).apply(lambda x: np.dot(x, range(1, 11)) / sum(range(1, 11)), raw=True)

    spx['MONTH'] = spx.index.month
    
    spx = spx.dropna(how='any')

    current_date = pd.Timestamp.now()
    # check if the last row's date is later than the current date
    if spx.index[-1] > current_date:
        spx = spx[:-1] # remove the last row, as it is incomplete

    print(f"Last rows {spx[-3:]}")
    
    num_features = spx.shape[1] + 1 # +1 because of the way we are splitting month
    
    return spx, num_features




def split_data(data, config):
    batch_size, seq_length, stride, step = config['batch_size'], config['seq_length'], config['n_days'], config['step']
    train_ratio= config['train_ratio']
    val_ratio = config['val_ratio']

    first_block_stride = seq_length + stride + (batch_size - 1) * step
    second_block_stride = stride + step * (batch_size - 1)
    rest_blocks_stride = batch_len = step * batch_size
    last_block_stride = batch_size * step - stride

    n_full_blocks = (len(data) - first_block_stride - second_block_stride * 5 - last_block_stride * 5) // rest_blocks_stride + 11

    
    new_length = first_block_stride + second_block_stride * 5 + last_block_stride * 5 + rest_blocks_stride * (n_full_blocks - 11)

    offset = len(data) - new_length
    data = data[-new_length:]

    n_blocks_train = round(n_full_blocks * train_ratio)
    n_blocks_val = round(n_full_blocks * val_ratio)
    n_blocks_test = n_full_blocks - n_blocks_train - n_blocks_val * 3
    
    idx_train = first_block_stride + second_block_stride + last_block_stride + rest_blocks_stride * (n_blocks_train - 3)
    idx_val_1 = idx_train + second_block_stride + last_block_stride + rest_blocks_stride * (n_blocks_val - 2)
    idx_val_2 = idx_val_1 + second_block_stride + last_block_stride + rest_blocks_stride * (n_blocks_val - 2)
    idx_val_3 = idx_val_2 + second_block_stride + last_block_stride + rest_blocks_stride * (n_blocks_val - 2)

    train_data = data[:idx_train]
    val_1_data = data[idx_train:idx_val_1]
    val_2_data = data[idx_val_1:idx_val_2]
    val_3_data = data[idx_val_2:idx_val_3]
    test_data = data[idx_val_3:]

    print(f"Train data shape {train_data.shape}, val_1_data shape {val_1_data.shape}, val_2_data shape {val_2_data.shape}, val_3_data shape {val_3_data.shape}, test_data shape {test_data.shape}")

    if (len(train_data) - first_block_stride - second_block_stride - last_block_stride) % rest_blocks_stride != 0:
        raise Exception("Training data not multiple")
    if (len(val_1_data) - second_block_stride - last_block_stride) % rest_blocks_stride != 0:
        raise Exception("First validation set data not multiple")
    if (len(val_2_data) - second_block_stride - last_block_stride) % rest_blocks_stride != 0:
        raise Exception("Second validation set data not multiple")
    if (len(val_3_data) - second_block_stride - last_block_stride) % rest_blocks_stride != 0:
        raise Exception("Third validation set data not multiple")
    if (len(test_data) - second_block_stride - last_block_stride) % rest_blocks_stride != 0:
        raise Exception("Test set data not multiple")

    print(f"SUCCESSFUL SPLIT!")
    data_sets = (train_data, val_1_data, val_2_data, val_3_data, test_data)

    return data_sets


def get_trainval_data_split(data, model_params, test_set):
    train_data, val_1_data, val_2_data, val_3_data, test_data = data

    if test_set == 'val_2':
        trainval_data = pd.concat((train_data, val_1_data))
        test_data = val_2_data
    elif test_set == 'val_3':
        trainval_data = pd.concat((train_data, val_1_data, val_2_data))
        test_data = val_3_data
    else:
        trainval_data = pd.concat((train_data, val_1_data, val_2_data, val_3_data))
    
    trainval_data = adjust_trainval_set(trainval_data, model_params)
    if model_params['variable_context_size'] is True:
        set_context_len(trainval_data, model_params)
    return trainval_data, test_data



scalers_dict = {'MinMaxScaler': MinMaxScaler((-1,1)),
               'StandardScaler': StandardScaler()}


from sklearn.preprocessing import PowerTransformer

from visualize import visualize_ranges

def undo_general_treatment(scaled_data, scaled_predictions, transformers):
    """
    scaled_predictions is a list or numpy array preferrably
    """    
    price_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'VWAP', 'WMA']#, 'ROC']
    volume_features = ['Volume']
    macro_features = ['VIX', 'DXY']
    month_feature = ['MONTH']

    general_treatment_columns = price_features + volume_features + macro_features # should be imported
    
    detrender, deseasonalizer = transformers

    original_index = scaled_data.index
    scaled_data = scaled_data.asfreq('W')
    scaled_data.index = pd.PeriodIndex(scaled_data.index, freq='W')

    scaled_predictions = scaled_predictions[1:] # ignore the day 0 predictions
    # better to add day -1, unscaled. it won't be used for anything other than operating prior to day 0
    # or use method onStart() inside the strategy to deal with it

    prediction_dfs = []

    for i in range(len(scaled_predictions[0])): # length should be one more
        start_date = scaled_data.index[i]
        new_index = pd.date_range(freq='W', start=start_date.start_time, periods=len(scaled_predictions))
        new_index = pd.PeriodIndex(new_index, freq='W')
        prediction_dfs.append(pd.DataFrame(data=scaled_predictions[:,i], index=new_index, columns=['Open']))

    filler = pd.DataFrame(data=np.zeros((len(scaled_predictions), len(general_treatment_columns))), columns=general_treatment_columns)

    for i in range(len(prediction_dfs)):
        filler.index = prediction_dfs[i].index
        filler['Open'] = prediction_dfs[i]['Open'].values
        if deseasonalizer is not None:
            filler = deseasonalizer.inverse_transform(filler)
        if detrender is not None:
            filler = detrender.inverse_transform(filler)
        filler = np.exp(filler)
        prediction_dfs[i]['Open'] = filler['Open'].values

    final_index = pd.date_range(freq='W', start=scaled_data.index[0].start_time, periods=len(scaled_predictions))
    # when we add the "extra" prediction, length will be the same so we can use original index
    result_df = pd.DataFrame(data=np.concatenate(([df['Open'].values.reshape(-1,1) for df in prediction_dfs]), axis=1), index=final_index) # columns=[f'Open_{i}' for i in range(prediction_dfs)]
    
    return result_df

# from visualize import visualize_monthly_data

def apply_general_treatment(train_df, val_df, treatment_config):
    """
    Normalize and preprocess the train DataFrame.
    
    train_df contains the following columns:
    Open, High, Low, Close, Adj Close, Volume, VIX, DXY, ROC, VWAP, WMA, MONTH
    """

    train_df = train_df.asfreq('W')
    val_df = val_df.asfreq('W')

    original_train_index = train_df.index
    original_val_index = val_df.index


    train_df.index = pd.PeriodIndex(train_df.index, freq='W')
    val_df.index = pd.PeriodIndex(val_df.index, freq='W')

    # nan_rows = train_df[train_df.isnull().any(axis=1)]
    # print(f"nan rows {nan_rows}")
    
    train_df = train_df.dropna(how='any') # then some weeks will be missing
    
    price_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'VWAP', 'WMA']#, 'ROC']
    volume_features = ['Volume']
    macro_features = ['VIX', 'DXY']
    month_feature = ['MONTH']

    general_treatment_columns = price_features + volume_features + macro_features

    if treatment_config['log'] == True:
        train_df[general_treatment_columns] = np.log(train_df[general_treatment_columns])
        val_df[general_treatment_columns] = np.log(val_df[general_treatment_columns])

    # print(f"Any null? {train_df.isnull().any().any()}")

    if treatment_config['remove_trend'] == True:
        detrender = Detrender() # trend degree?
        train_df[general_treatment_columns] = detrender.fit_transform(train_df[general_treatment_columns])
        val_df[general_treatment_columns] = detrender.transform(val_df[general_treatment_columns])
    else:
        detrender = None
    

    if treatment_config['remove_seasonality'] == True:
        deseasonalizer = Deseasonalizer(sp=52, model='additive')
        train_df[general_treatment_columns] = deseasonalizer.fit_transform(train_df[general_treatment_columns])
        val_df[general_treatment_columns] = deseasonalizer.transform(val_df[general_treatment_columns])
    else:
        deseasonalizer = None

    train_df['Month_Sin'] = np.sin(2 * np.pi * train_df['MONTH'] / 12)
    train_df['Month_Cos'] = np.cos(2 * np.pi * train_df['MONTH'] / 12)
    train_df = train_df.drop(columns=month_feature)  # drop the original MONTH column
    val_df['Month_Sin'] = np.sin(2 * np.pi * val_df['MONTH'] / 12)
    val_df['Month_Cos'] = np.cos(2 * np.pi * val_df['MONTH'] / 12)
    val_df = val_df.drop(columns=month_feature)

    
    train_df = train_df.set_index(original_train_index)
    val_df = val_df.set_index(original_val_index)

    # visualize_ranges(train_df, train_df.columns)
    
    return train_df, val_df, detrender, deseasonalizer


transformers_dict = {'StandardScaler': StandardScaler,
                     'MinMaxScaler': MinMaxScaler,
                     'PowerTransformer': PowerTransformer # yeo johnson?
                    }

import os


def create_overlapping_targets(data, column, window_size, step):
    targets = []
    for i in range(0, len(data) - window_size + 1, step):
        targets.append(data[column].iloc[i:i+window_size].values.tolist())
    targets = pd.DataFrame(targets, index=data.index[:len(targets)])
    return targets.values.tolist()

                     

def apply_normalization(data, transformer_name=None, detrender=None, transformer=None): # dynamic treatment
    """
    either transformer name exists or both instances of detrender and transformer exist
    """
    price_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'VWAP', 'WMA']#, 'ROC']
    volume_features = ['Volume']
    macro_features = ['VIX', 'DXY']
    
    general_treatment_cols = list(set(price_features + volume_features + macro_features))

    scaled_data = data.copy()

    if detrender is None or transformer is None:
        detrender = Detrender()
        scaled_data[general_treatment_cols] = detrender.fit_transform(scaled_data[general_treatment_cols].values)
        transformer = transformers_dict[transformer_name]()
        scaled_data[general_treatment_cols] = transformer.fit_transform(scaled_data[general_treatment_cols].values) # one final transformation to keep similar/equal range for all features
        return scaled_data, detrender, transformer
    
    # visualize_ranges(scaled_data[general_treatment_cols], column_labels=general_treatment_cols, title="old_distribution.png")
    scaled_data[general_treatment_cols] = detrender.transform(scaled_data[general_treatment_cols].values)
    scaled_data[general_treatment_cols] = transformer.transform(scaled_data[general_treatment_cols].values)
    # visualize_ranges(scaled_data[general_treatment_cols], column_labels=general_treatment_cols, title="new_distribution.png")
    return scaled_data
    
    
def recover_dynamic_data(context, prediction, transformer, detrender):
    price_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'VWAP', 'WMA']#, 'ROC']
    volume_features = ['Volume']
    macro_features = ['VIX', 'DXY']
    ## problem with the order of the columns. solved?
    general_treatment_cols = list(set(price_features + volume_features + macro_features))

    filled_prediction = pd.DataFrame(data=[], columns=general_treatment_cols) # this is only for batch size 1. i think it's good, for sequence
    filled_prediction['Open'] = prediction.reshape(-1)
    
    for col in general_treatment_cols:
        if col != 'Open':
            filled_prediction[col] = np.zeros_like(prediction).reshape(-1)
        
    data = pd.concat((context[general_treatment_cols], filled_prediction))
    
    data[general_treatment_cols] = transformer.inverse_transform(data[general_treatment_cols].values)
    data[general_treatment_cols] = detrender.inverse_transform(data[general_treatment_cols].values)
    return data[['Open']].values ## it might just be better to use all columns and not differentiate nor need to name them
    
    
def recover_original_prediction(scaled_prediction, contexts, detrenders, transformers, batch_size, stride):
    original_prediction = np.zeros_like(scaled_prediction)
    for i in range(batch_size):
        original_prediction[i] = recover_dynamic_data(contexts[i], scaled_prediction[i], transformers[i], detrenders[i])[-stride:].flatten()
    return original_prediction.tolist()


def adjust_trainval_set(trainval_data, model_config):
    context_len, batch_size, seq_length, stride, step = model_config['context_len'], model_config['batch_size'], model_config['seq_length'], model_config['n_days'], model_config['step']

    first_block_size = seq_length + stride + (batch_size - 1) * step
    second_block_size = stride + step * (batch_size - 1)
    rest_block_size = batch_len = step * batch_size
    last_block_size = batch_size * step - stride

    
    to_substract = 0
    while (len(trainval_data) - to_substract - first_block_size - second_block_size - last_block_size) % rest_block_size != 0:
        to_substract += 1
    
    return trainval_data[to_substract:]
    
    

def set_context_len(train_data, config):
    factor, batch_size, seq_length, stride, step = config['context_factor'], config['batch_size'], config['seq_length'], config['n_days'], config['step']
    
    first_block_size = seq_length + stride + (batch_size - 1) * step
    second_block_size = stride + step * (batch_size - 1)
    rest_block_size = batch_len = step * batch_size
    last_block_size = batch_size * step - stride

    """factor must be > 1 """
    
    context_len = len(train_data) // factor 
    if context_len < first_block_size:
        context_len = first_block_size
    else:
        while (context_len-first_block_size) % rest_block_size != 0:
            context_len += 1
            
    config['context_len'] = context_len # does it get updated globally?
    return
    

def check_split(len_context, len_following, config, is_train):
    factor, batch_size, seq_length, stride, step = config['context_factor'], config['batch_size'], config['seq_length'], config['n_days'], config['step']

    first_block_size = seq_length + stride + step * (batch_size - 1)
    second_block_size = stride + step * (batch_size - 1)
    rest_block_size = batch_len = step * batch_size
    last_block_size = batch_size * step - stride

    print(f"\nLen of context {len_context}, len of following {len_following}, is train {is_train}, first block size {first_block_size}, second block size {second_block_size}, rest block size {rest_block_size}")

    if (len_context - first_block_size) % rest_block_size != 0:
        raise Error("Context not multiple")
            
    if (len_following - second_block_size - last_block_size) % rest_block_size != 0:
        raise Error("Following not multiple")

    return
        
    

def split_set(config, context_len, previous_data, data):
    print(f"Full data len {len(data)}")
    if previous_data is None:
        context = data[:context_len]
        following = data[context_len:]
    else:
        context = previous_data[-context_len:]
        following = data
        
    return (context, following)

    
def get_dataloader(previous_data, data, config, is_train, extra=False):
    context, following = split_set(config, config['context_len'], previous_data, data)
    check_split(len(context), len(following), config, is_train)
    
    dataset = StockMultibatchDataset(config, context, following, config['step'], 'multifeature', extra)
    original_length = dataset.original_length
    dataloader = DataLoader(dataset, config['batch_size'], shuffle=False, collate_fn = custom_collate_fn)
    return dataloader, original_length
    
    
class StockDataset(Dataset): # dealing with batch size correctly?
    def __init__(self, config, data):
        self.seq_length = config['seq_length']
        self.stride = config['n_days']
        self.data = data

    def __len__(self):
        length = (len(self.data) - self.seq_length) // self.stride
        return length

    def __getitem__(self, index):
        start = index * self.stride
        x = self.data[start:start+self.seq_length, :]
        y = self.data[start+self.seq_length:start+self.seq_length+self.stride, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class Stock1Dataset(Dataset):
    def __init__(self, config, data):
        self.seq_length = config['seq_length']
        self.stride = config['n_days']
        self.data = data

    def __len__(self):
        length = (len(self.data) - self.seq_length) // self.stride
        return length

    def __getitem__(self, index):
        start = index * self.stride
        x = self.data[start:start+self.seq_length, 0]
        y = self.data[start+self.seq_length:start+self.seq_length+self.stride, 0]
        return torch.tensor(x, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.float32).to(device)

class StockMultibatchDataset(Dataset): # when there is not enough data, is creates smaller batches that won't fit the model instead of throwing an error. Change it
    def __init__(self, config, initial_context, following_data, step, type='multifeature', extra=False):
        self.config = config
        self.batch_size = config['batch_size']
        self.seq_length = config['seq_length']
        self.stride = config['n_days']
        self.data = following_data
        self.context = initial_context
        self.context_len = len(initial_context)
        self.step = step
        self.type = type
        self.extra = extra
        self.original_length = (len(self.data) - self.stride + self.step) // (self.step * self.batch_size)

    def __len__(self):
        length = (len(self.data) - self.stride + self.step) // self.step
        if self.extra == True:
            length += self.stride // self.step # sequences, not batches?
        return length

    def __getitem__(self, index):
        start = (index - 1) * self.step
        full_context = pd.concat((self.context, self.data[:start+self.step])).iloc[-self.context_len:]#[-self.context_len - index % self.batch_size:]
        scaled_full_context, input_detrender, input_transformer = apply_normalization(full_context, self.config['normalization']['transformer']) # dynamic
        if self.type == 'multifeature':
            og_x = torch.tensor(full_context.values[-self.seq_length:], dtype=torch.float32).to(device)
            scaled_x = torch.tensor(scaled_full_context.values[-self.seq_length:], dtype=torch.float32).to(device)
        else:
            og_x = torch.tensor(full_context[['Open']].values[-self.seq_length:], dtype=torch.float32).to(device)
            scaled_x = torch.tensor(scaled_full_context[['Open']].values[-self.seq_length:], dtype=torch.float32).to(device) # tensors are slow

        if index < self.original_length * self.batch_size:
            target_context = pd.concat((self.context, self.data[:start+self.step+self.stride])).iloc[-self.context_len+self.stride:]#[-self.context_len - index % self.batch_size - self.stride:] doubts on size
            scaled_target_context = apply_normalization(target_context, self.config['normalization']['transformer'], input_detrender, input_transformer)
            og_y = target_context[['Open']].values[-self.stride:].reshape(-1).tolist()
            scaled_y = torch.tensor(scaled_target_context[['Open']].values[-self.stride:].reshape(-1), dtype=torch.float32).to(device)
        else:
            og_y = scaled_y = torch.tensor([])
        
        return og_x, og_y, scaled_x, scaled_y, scaled_full_context, input_detrender, input_transformer

def custom_collate_fn(batch):
    x, y, scaled_x, scaled_y, scaled_full_context, detrender, transformer = zip(*batch)
    return (
        torch.stack(x),
        list(y),
        torch.stack(scaled_x),
        torch.stack(scaled_y),
        scaled_full_context,
        detrender,
        transformer
    )
