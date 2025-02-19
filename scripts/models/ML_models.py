import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.models_utils import calculate_loss
from data import get_dataloader, recover_original_prediction
from time import sleep
import torch.optim as optim


from device import device

class AutoRegressiveNN(nn.Module): # it usually requires a higher Learning rate than DL models
    def __init__(self, input_size=15, hidden_size=32, output_size=2):
        super(AutoRegressiveNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x[:, -1, :]
        return x


class ARNNModel:
    def __init__(self, config):#input_size, hidden_size=32, learning_rate=0.001):
        self.input_size = config['num_features']
        self.hidden_size = config['hidden_units']
        self.model = AutoRegressiveNN(self.input_size, self.hidden_size, config['n_days']).to(device)
        self.optimizer = set_optimizer(self.model, config)
        self.criterion = nn.MSELoss()
        self.future_steps = config['n_days']
        self.config = config # for what?


    def predict(self, data, future_steps=1): # useful for real time predictions
        scaled_context, scalers, detrenders = apply_normalization(data, self.config) # should include some 1d normalization option
        final_input = torch.tensor(data[-self.input_size:, 0], dtype=torch.float32).to(device)
        # self.model.eval()
        prediction = self.model(final_input)#.item()
        # predictions = np.array(predictions)
        return scaled_context, prediction, scalers[0], detrenders[0]# self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


   
    def train_validate_forward_dataloader(self, train_data, num_epochs=None):
        self.model.train()

        train_dataloader, original_length = get_dataloader(None, train_data, self.config, is_train=True, extra=True)
        
        train_targets, train_predictions, losses_by_epochs = [], [], []
        train_loss = 0
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        for k in range(num_epochs):
            epoch_loss = 0
            i = 0
            for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in train_dataloader:
                if i >= original_length:
                    continue
                scaled_prediction = self.model(scaled_inputs)
                loss = calculate_loss(scaled_targets, scaled_prediction, self.config)
                
                self.optimizer.zero_grad()
                loss.backward()
                # clipping?
                self.optimizer.step()
                epoch_loss += loss
                i+=1
                
            epoch_loss /= len(train_dataloader)

        return losses_by_epochs
    
    def validate_forward_dataloader(self, train_data, val_data):
        self.model.eval()

        val_dataloader, original_length = get_dataloader(train_data, val_data, self.config, is_train=False, extra=True)
        
        val_scaled_targets, val_scaled_predictions, extra_scaled_predictions = [], [], []
        val_log_targets, val_log_predictions, extra_log_predictions = [], [], []
        
        val_loss, k = 0, 0
        
        with torch.no_grad():
            for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in val_dataloader:
                scaled_prediction = self.model(scaled_inputs)
                common_scale_prediction = recover_original_prediction(scaled_prediction.cpu().numpy(), scaled_contexts, detrenders, transformers, self.config['batch_size'], self.config['n_days']) 
                
                if k < original_length:
                    val_scaled_predictions.extend(scaled_prediction.cpu().numpy().tolist())
                    val_scaled_targets.extend(scaled_targets.cpu().numpy().tolist())
                    val_log_predictions.extend(common_scale_prediction)
                    val_log_targets.extend(targets)
                else:
                    extra_scaled_predictions.extend(scaled_prediction.cpu().numpy().tolist())
                    extra_log_predictions.extend(common_scale_prediction)

                k += 1

            print(f"Val targets shape {np.array(val_log_targets).shape}, predictions shape {np.array(val_log_predictions).shape}")
            val_loss = calculate_loss(val_log_targets, val_log_predictions, self.config)

            val_scaled_predictions.extend(extra_scaled_predictions)
            val_log_predictions.extend(extra_log_predictions)

            return np.array(val_log_predictions), np.array(val_log_targets), np.array(val_scaled_predictions), np.array(val_scaled_targets), val_loss



from xgboost import XGBRegressor


class XGBoostModel(): # one-dimensional?
    def __init__(self, config):
        xgboost_params = config['xgboost']
        self.model = XGBRegressor(
            n_estimators=xgboost_params.get("n_estimators", 200),
            max_depth=xgboost_params.get("max_depth", 10),
            learning_rate=xgboost_params.get("learning_rate_x", 0.01),
            objective=xgboost_params.get("objective", "reg:squarederror"),
            random_state=xgboost_params.get("random_state", 42),
            verbosity=xgboost_params.get("verbosity", 0),
            tree_method='hist'
        )
        self.config = config


    def train_validate_forward_dataloader(self, train_data, num_epochs=None): # how many epochs?? works with batch_Size > 1???
        train_dataloader, original_length = get_dataloader(None, train_data, self.config, is_train=True, extra=True)

        all_inputs, all_targets = [], []

        i = 0
        for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in train_dataloader:
            if i>=original_length:
                continue
            all_inputs.extend(scaled_inputs.cpu().tolist())
            all_targets.extend(scaled_targets.cpu().tolist())
            i += 1

        all_inputs = np.array(all_inputs).reshape(len(all_inputs), -1).tolist() # wouldnt it train better with a numpy array?

        self.model.fit(all_inputs, all_targets)
        print("XGBoost training completed")
        return

    def validate_forward(self, initial_context, following_data):
        context = initial_context
        predictions, targets = [], []
        for i in range(len(following_data)):
            prediction = self.model.predict(context.reshape(1,-1))[0]
            predictions.append(prediction)
            context = np.append(context[1:], following_data[i:i+1])
            targets.append(context[-1])
            loss += calculate_loss(predictions, targets)
        loss = loss_fn(np.array(predictions), np.array(targets))
        return predictions, targets, loss

    def validate_forward_dataloader(self, train_data, val_data):
        val_dataloader, original_length = get_dataloader(train_data, val_data, self.config, is_train=False, extra=True)
        
        batch_size, stride = self.config['batch_size'], self.config['n_days']
        
        val_predictions, val_targets, extra_predictions = [], [], []
        val_scaled_predictions, val_scaled_targets, extra_scaled_predictions = [], [], []

        input_counter = 0
        
        k=0
        for inputs, targets, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in val_dataloader:
            scaled_prediction = np.zeros((batch_size, stride))
            for i in range(batch_size):
                input_counter += 1
                scaled_prediction[i] = self.model.predict(scaled_inputs[i].reshape(1, -1))
            common_scale_prediction = recover_original_prediction(scaled_prediction, scaled_contexts, detrenders, transformers, batch_size, stride)
            if k < original_length:
                val_scaled_targets.extend(scaled_targets.cpu().numpy().tolist())   
                val_scaled_predictions.extend(scaled_prediction.tolist())
                val_targets.extend(targets)   
                val_predictions.extend(common_scale_prediction)
            else:
                extra_scaled_predictions.extend(scaled_prediction.tolist())
                extra_predictions.extend(common_scale_prediction)
            k += 1

                
        val_loss = calculate_loss(val_targets, val_predictions, self.config) # maybe better calculate it each step, this way it seems to small

        val_predictions.extend(extra_predictions)
        val_scaled_predictions.extend(extra_scaled_predictions)

        return np.array(val_predictions), np.array(val_targets), np.array(val_scaled_predictions), np.array(val_scaled_targets), val_loss

    
def set_optimizer(model, params): # make it modular better
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    return optimizer


ML_models_dict = {'ARNN': ARNNModel, 'XGBoost': XGBoostModel}

def get_ML_model(config): # should make it easy to decide which ones to use. model-dependant
    model_name = config['model']
    model_class = ML_models_dict[model_name]
    model = model_class(config=config)
    return model