import torch
import torch.nn as nn

import torch.optim as optim

from data import *
from visualize import *

import warnings
warnings.simplefilter("ignore", UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)

criterion_dict = { # duplicated. put it in another module
    'MSELoss': torch.nn.MSELoss(),
    'RMSELoss': torch.nn.MSELoss(),
    'L1Loss': torch.nn.L1Loss(),
    'SmoothL1Loss': torch.nn.SmoothL1Loss(),
    'HuberLoss': torch.nn.HuberLoss()
}

def calculate_loss(prediction, target, config):
    if type(prediction) is not torch.Tensor:
        prediction = torch.tensor(prediction, dtype=torch.float32).to(device)
    if type(target) is not torch.Tensor:
        target = torch.tensor(target, dtype=torch.float32).to(device)
        
    criterion = criterion_dict[config['loss_function']]
    loss = criterion(prediction, target)
    # print(f"Loss... {loss}")
    if config['loss_function'] == 'RMSELoss':
        loss = torch.sqrt(loss)
    return loss


def set_optimizer(model, params):
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    return optimizer

def get_initial_context(train_data, config, step, is_train): # maybe don't need to check multiplicity, already checked in split_data()
    print(f"Getting initial context...")
    batch_size, seq_length, stride = config['batch_size'], config['seq_length'], config['n_days']
    
    first_block_size = seq_length + step * (batch_size-1) # cuánto avanzar para tener información suficiente para poder hacer la primera predicción
    if len(train_data) < first_block_size:
        raise Error('not enough data')
    second_block_size = stride + step * (batch_size - 1) # primera predicción, no hay overlap en la primera secuencia
    rest_block_size = batch_size * step # resto de predicciones, hay overlap en todas las secuencias

    context_len_top  = len(train_data) // config['context_factor'] # upper or lower bound?
    context_len_bottom = 0

    while (len(train_data)-context_len_top-second_block_size) % rest_block_size != 0:
        context_len_top += 1
    rest_data, rest_rest_data = len(train_data)-context_len_top, len(train_data)-context_len_top-first_block_size-second_block_size
    context_len_bottom = context_len_top
    print(first_block_size, second_block_size)
    while (context_len_bottom-first_block_size-second_block_size) % rest_block_size != 0:
        context_len_bottom -= 1
    to_substract = context_len_top - context_len_bottom

    if is_train == True:
        initial_context = train_data[to_substract:context_len_top]
    else:
        initial_context = train_data[-context_len_bottom:]
    
    return initial_context, to_substract # maybe return data following instead of substract


ticker_list = ['SPY', '^N225', 'HSI'] # diferentes contextos de mercado

def train_model(model_name, model, params, train_data, validate_train=False, set="train"): # maybe this function becomes unnecesary
    print(f"Training model {model_name}")
    train_epoch_losses = model.train_validate_forward_dataloader(train_data) # it should be like this for every model

    if validate_train is True:
        train_predictions, train_targets, train_scaled_predictions, train_scaled_targets, train_loss = model.validate_forward_dataloader(None, train_data)
        visualize_predictions(train_targets, train_predictions, train_loss, model_name, set)
        
    return model