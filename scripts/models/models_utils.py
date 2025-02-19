import torch
import torch.nn as nn

import torch.optim as optim

from data import *
from visualize import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

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


ticker_list = ['SPY', '^N225', 'HSI'] # diferentes contextos de mercado

def train_model(model_name, model, params, train_data, validate_train=False, set="train"): # maybe this function becomes unnecesary
    print(f"Training model {model_name}")
    train_epoch_losses = model.train_validate_forward_dataloader(train_data) # it should be like this for every model

    if validate_train is True:
        train_predictions, train_targets, train_scaled_predictions, train_scaled_targets, train_loss = model.validate_forward_dataloader(None, train_data)
        visualize_predictions(train_targets, train_predictions, train_loss, model_name, set)
        
    return model