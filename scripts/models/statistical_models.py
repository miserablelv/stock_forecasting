from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from models.models_utils import calculate_loss
from data import recover_original_prediction, get_dataloader
from time import sleep
# import torch
import warnings


class SARIMA:
    def __init__(self, order, seasonal_order, trend, config):
        """
        order params are (p, d, q)
        seasonal_order params (P, D, Q, m)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.n_steps = config['n_days']
        self.config = config
        warnings.filterwarnings("ignore", message="Non-invertible starting seasonal moving average")
        warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to")
        warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found.")


    def train(self, scaled_data):
        scaled_data = scaled_data.dropna(how='any')
        self.model = SARIMAX(endog=scaled_data['Open'].values, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend, enforce_stationarity=False) # enforce_invertibility? using exogenous variables?
        results = self.model.fit(disp=False)

        return results        

    def validate(self, val_data, config):
        predictions = self.model.forecast(steps=len(val_data))
        targets = val_data.values

        loss = np.mean((predictions - targets) ** 2)  # MSE
        return predictions, targets, loss

    def validate_forward(self, full_data):
        data, train_data, val_data = full_data
        context = get_initial_context()
        self.model.forecast(steps=self.config['n_days'])

        val_predictions, val_targets = [], []
        val_loss = 0
        
        for i in range(len(val_data)):
            self.train(context)
            scaled_target, target, context = update_context()
            val_targets.append(target)
            prediction = self.model.forecast(steps=self.config['n_days'])
            val_predictions.append(predictions)
            val_loss += np.mean((prediction - scaled_target) ** 2)

        return val_predictions, val_targets, val_loss

    def validate_forward(self, train_data, val_data):
        predictions, targets = [], []
        for i in range(len(val_data)):
            context = pd.concat((train_data, val_data.iloc[:i]))
            sarima_result = self.train(context)
            prediction = sarima_result.forecast(self.n_steps)
            predictions.extend(prediction)
            if i < len(val_data) - self.n_steps:
                targets.extend(val_data[i:i+self.n_steps])
                
        loss = calculate_loss(torch.tensor(predictions, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), self.config)
        return np.array(predictions), np.array(targets), loss
    
    def validate_forward_dataloader(self, train_data, val_data):
        val_dataloader, original_length = get_dataloader(train_data, val_data, self.config, is_train=False, extra=True)

        log_targets, log_predictions, extra_log_predictions = [], [], []
        scaled_targets, scaled_predictions, extra_scaled_predictions = [], [], []
        k = 0
        for input, target, scaled_input, scaled_target, scaled_contexts, detrenders, transformers in val_dataloader:
            scaled_prediction = []
            val_target = []
            for i in range(self.config['batch_size']):
                sarima_result = self.train(scaled_contexts[i])
                scaled_prediction.append(sarima_result.forecast(self.n_steps).tolist())
            common_scale_prediction = recover_original_prediction(np.array(scaled_prediction), scaled_contexts, detrenders, transformers, self.config['batch_size'], self.config['n_days'])
            if k < original_length:
                scaled_predictions.extend(scaled_prediction)
                log_predictions.extend(common_scale_prediction)
                log_targets.extend(target)
                scaled_targets.extend(scaled_target.cpu().numpy().tolist()) # i don't think targets are really necessary to return scaled but ok
            else:
                extra_log_predictions.extend(common_scale_prediction)
                extra_scaled_predictions.extend(scaled_prediction)
                # we'll have to recover in common scale too
            k += 1
            if k % 10 == 0:
                print(f"Validating SARIMA... {k}")

        loss = calculate_loss(log_predictions, log_targets, self.config)
        # print(f"SARIMA val loss {loss}")

        log_predictions.extend(extra_log_predictions)
        scaled_predictions.extend(extra_scaled_predictions)

        return np.array(log_predictions), np.array(log_targets), np.array(scaled_predictions), np.array(scaled_targets), loss
            

    def update_context(self, new_data):
        """
        Update model's internal state with new observations.
        :param new_data: Pandas Series or array of new values.
        """
        self.results = self.results.append(new_data)


def get_statistical_model(config):
    sarima_config = config['sarima']
    order = sarima_config['p_param'], sarima_config['d_param'], sarima_config['q_param']
    s_order = sarima_config['P_param'], sarima_config['D_param'], sarima_config['Q_param'], sarima_config['m_param']
    trend = sarima_config['t_param']
    model = SARIMA(order, s_order, trend, config)
    return model