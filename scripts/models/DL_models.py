import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import sys
import os

from models.models_utils import calculate_loss, train_model
from data import recover_original_prediction, get_dataloader

from torch.amp import GradScaler, autocast # speed up in exchange for a small loss in precision

from time import sleep

# necessary?
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

criterion_dict = {
    'MSELoss': torch.nn.MSELoss(),
    'RMSELoss': torch.nn.MSELoss(),
    'L1Loss': torch.nn.L1Loss(),
    'SmoothL1Loss': torch.nn.SmoothL1Loss(),
    'HuberLoss': torch.nn.HuberLoss()
}

class StockRNNBase(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_layers, output_dim, dropout, config):
        super(StockRNNBase, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.rnn_type = rnn_type
        
        self.rnn = rnn_type(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.config = config

        self.hidden = None

        self.epochs_trained = 0

        self.to(device)

    def forward(self, x, hidden=None):
        if x.shape[2] != self.config['num_features']:
            print("\n\nNum features not according\n\n")
            print(x.shape)
        if x.shape[0] != self.config['batch_size']:
            print("Check batch size modification")
            print(f"Theoretical size {self.config['batch_size']}, real batch size {x.shape[0]}")
            exit(0)
        out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return hidden, out

    def init_hidden(self, batch_size):
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return h0

    def detach_hidden_state(self, hidden):
        if type(hidden) is tuple:
            return tuple([h.detach() for h in hidden])
        else:
            return hidden.detach()


    def train_validate_forward_dataloader(self, train_data, num_epochs=None):
        self.train()
        train_dataloader, original_length = get_dataloader(None, train_data, self.config, is_train=True, extra=True)
        criterion, optimizer = criterion_dict[self.config['loss_function']], set_optimizer(self, self.config)
        train_losses = []
        torch.autograd.set_detect_anomaly(True)
        hidden = None
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        else:
            print("Resuming training for model...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            hidden = self.init_hidden(self.config['batch_size'])
            k = 0
            for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in train_dataloader:
                if k >= original_length: # the last ones don't have target
                    continue
                if self.config['prime_hidden'] == True:
                    hidden = self.prime_hidden() # initialize and detach inside function?
                else: # keep hidden state
                    hidden = self.detach_hidden_state(hidden)
                hidden, scaled_prediction = self(scaled_inputs, hidden)

                loss = calculate_loss(scaled_prediction, scaled_targets, self.config)
                optimizer.zero_grad()
                loss.backward() # retain_graph=True
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config['gradient_clipping'])  # Gradient clipping
                optimizer.step()
                epoch_loss += loss.item() ## IF YOU ARE GOING TO ADD IT ON EACH TIMESTEP, CRITERION MAYBE SHOULD BE SET AS 'SUM' AND NOT 'MEAN'
                k+=1
                
            epoch_loss /= len(train_dataloader)
            train_losses.append(epoch_loss)
            self.epochs_trained += 1
            print(f"Epoch {self.epochs_trained}. Loss {epoch_loss}")
        self.hidden = hidden # we can keep it to use it as a starter later in validation
        return train_losses
    
    def validate_forward_dataloader(self, train_data, val_data):
        self.eval()
        val_targets, val_predictions, extra_predictions = [], [], []
        scaled_val_targets, scaled_val_predictions, scaled_extra_predictions = [], [], []
        val_loss = 0
        k = 0

        val_dataloader, original_length = get_dataloader(train_data, val_data, self.config, is_train=False, extra=True)
        
        with torch.no_grad():
            for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in val_dataloader:
                self.hidden = self.detach_hidden_state(self.hidden)
                self.hidden, scaled_prediction = self(scaled_inputs, self.hidden)
                common_scale_prediction = recover_original_prediction(scaled_prediction.cpu().numpy(), scaled_contexts, detrenders, transformers, self.config['batch_size'], self.config['n_days']) 
                if k < original_length:
                    val_predictions.extend(common_scale_prediction)
                    scaled_val_predictions.extend(scaled_prediction.cpu().tolist())
                    scaled_val_targets.extend(scaled_targets.cpu().tolist())
                    val_targets.extend(targets)
                else:
                    extra_predictions.extend(common_scale_prediction)
                    scaled_extra_predictions.extend(scaled_prediction.cpu().tolist())
                k += 1
        val_loss = calculate_loss(val_predictions, val_targets, self.config)
        val_predictions.extend(extra_predictions)
        scaled_val_predictions.extend(scaled_extra_predictions)

        return np.array(val_predictions), np.array(val_targets), np.array(scaled_val_predictions), np.array(scaled_val_targets), val_loss
    
    def prime_hidden(self, scaled_context): # revisit
        dataset = StockDataset(self.config, scaled_context)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        hidden = self.init_hidden(self.config['batch_size'])
        for inputs, targets, scaled_inputs, scaled_targets, contexts in dataloader: # this dataloader does not fit right now?
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = self.detach_hidden_state(hidden)
            with torch.autocast(device_type="cuda"):
                hidden, _ = self(inputs, hidden)
        return hidden

    def future_predictions(self, context, hidden=None):
        batch_size, seq_length, stride, num_features = self.config['batch_size'], self.config['seq_length'], self.config['n_days'], self.config['num_features']
        
        hidden = self.detach_hidden_state(hidden)
        with torch.autocast(device_type="cuda"):
            hidden, outputs = self(final_input, hidden)
        future_prediction = outputs.reshape(-1)[-stride:]
        
        return future_prediction, hidden

    def get_prediction(self, context, hidden): # using batch size 1 for real time predictions
        context, scaler, detrender, deseasonalizer = apply_normalization(context, self.config)

        if hidden is None:
            hidden = self.prime_hidden(context)

        batch_size, seq_length, stride, num_features = self.config['batch_size'], self.config['seq_length'], self.config['n_days'], self.config['num_features']

        final_input = torch.tensor(context[-seq_length:].reshape((batch_size, seq_length, num_features)), dtype=torch.float32).to(device)
        hidden = self.detach_hidden_state(hidden)
        hidden, outputs = self(final_input, hidden)

        prediction = outputs.reshape(-1)[-stride:]
        
        return hidden, context, prediction, scaler, detrender, deseasonalizer


class StockRNN(StockRNNBase):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, config):
        super(StockRNN, self).__init__(nn.RNN, input_dim, hidden_dim, num_layers, output_dim, dropout, config)

class StockLSTM(StockRNNBase):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, config):
        super(StockLSTM, self).__init__(nn.LSTM, input_dim, hidden_dim, num_layers, output_dim, dropout, config)

class StockGRU(StockRNNBase):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, config):
        super(StockGRU, self).__init__(nn.GRU, input_dim, hidden_dim, num_layers, output_dim, dropout, config)

models_dict = {'RNN': StockRNN, 'LSTM': StockLSTM, 'GRU': StockGRU}

import torch
import torch.nn as nn

class StockCNN(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, output_dim, dropout, config):
        super(StockCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, output_dim)
        self.config = config
        self.to(device)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, input_channels, sequence_length]
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x.mean(dim=-1))  # pooling
        return x

    def train_on_data(self, dataloader, config):
        self.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=config['lr']) # no other optimizers?
        for epoch in range(config['num_epochs']):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def train_validate_forward_dataloader(self, train_data, num_epochs=None):
        self.train()
        train_dataloader, original_length = get_dataloader(None, train_data, self.config, is_train=True, extra=True)
        criterion, optimizer = criterion_dict[self.config['loss_function']], set_optimizer(self, self.config)
        losses_by_epoch = []
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        for epoch in range(self.config['num_epochs']):
            epoch_loss = 0
            k = 0
            for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in train_dataloader:
                if k >= original_length:
                    continue
                scaled_prediction = self.forward(inputs)
                loss = calculate_loss(scaled_prediction, scaled_targets, self.config)
                # exploding gradients?
                # vanishing gradients?
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                k += 1
            epoch_loss /= len(train_dataloader)
            losses_by_epoch.append(epoch_loss)
        return losses_by_epoch

    def validate_forward_dataloader(self, train_data, val_data):
        self.eval()
        val_log_targets, val_log_predictions, extra_log_predictions = [], [], []
        val_scaled_targets, val_scaled_predictions, extra_scaled_predictions = [], [], []
        val_dataloader, original_length = get_dataloader(train_data, val_data, self.config, is_train=False, extra=True)
        val_loss = 0
        k = 0

        with torch.no_grad():
            for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in val_dataloader:
                scaled_prediction = self.forward(scaled_inputs)
                common_scale_prediction = recover_original_prediction(scaled_prediction.cpu().numpy(), scaled_contexts, detrenders, transformers, self.config['batch_size'], self.config['n_days'])
                if k < original_length:
                    val_log_targets.extend(targets)
                    val_log_predictions.extend(common_scale_prediction)
                    val_scaled_targets.extend(scaled_targets.cpu().tolist())
                    val_scaled_predictions.extend(scaled_prediction.cpu().tolist())
                else:
                    extra_log_predictions.extend(common_scale_prediction)
                    extra_scaled_predictions.extend(scaled_prediction.cpu().tolist())
                k += 1
            val_loss = calculate_loss(torch.tensor(val_log_predictions, dtype=torch.float32).to(device), torch.tensor(val_log_targets, dtype=torch.float32).to(device), self.config)
            
        val_log_predictions.extend(extra_log_predictions)
        val_scaled_predictions.extend(extra_scaled_predictions)
        
        return np.array(val_log_predictions), np.array(val_log_targets), np.array(val_scaled_predictions), np.array(val_scaled_targets), val_loss


def set_optimizer(model, params): # model independent
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    return optimizer

def get_DL_model(config): # should make it easy to decide which ones to use
    if config['model'] != 'CNN':
        model_class = models_dict[config['model']]
        model = model_class(input_dim=config['num_features'], hidden_dim=config['hidden_units'], num_layers=config['num_layers'], output_dim=config['n_days'], dropout=config['dropout'], config=config)
    else:
        model = StockCNN(input_dim=config['num_features'], num_filters=config['num_filters'], kernel_size=config['kernel_size'], output_dim=config['n_days'], dropout=config['dropout'], config=config) # necessary to pass the whole dictionary at the end?
    return model