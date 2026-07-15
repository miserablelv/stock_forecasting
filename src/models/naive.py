import numpy as np
from data.loader import get_dataloader
from models.utils import recover_original_prediction, calculate_loss
import torch



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def drift_model(series, steps_ahead):
    """
    Drift model for time series forecasting.
    It calculates the average daily change in price and expects it to remain that way
    
    Parameters:
    - series: The time series data as a pandas Series.
    - steps_ahead: Number of steps to forecast into the future.
    
    Returns:
    - Forecasted values as a pandas Series.
    """
    n = len(series)
    slope = (series.iloc[-1] - series.iloc[0]) / (n - 1)
    forecast = [series.iloc[-1] + slope * i for i in range(1, steps_ahead + 1)]
    return pd.Series(forecast, index=range(n, n + steps_ahead))

# Forecast 5 steps ahead
# forecast = drift_model(time_series, 5)

# # Plot original data and forecast
# plt.plot(time_series, label="Actual Data", marker='o')
# plt.plot(forecast, label="Forecast", marker='o', linestyle='dashed')
# plt.legend()
# plt.title("Drift Model Forecast")
# plt.show()

# this one is also univariate
class LastKMedian:
    def __init__(self, config):
        """
        Initialize the Last-K Median model.
        :param k: The number of most recent data points to use for calculating the median.
        """
        self.k = config['k_median']['k']
        self.history = []
        self.config = config
        self.config['batch_size']=1
        self.config['n_days']=1

    def train(self, data):
        """
        Fit the Last-K Median model to the data.
        :param data: List or array of values.
        """
        self.history = list(data)

    def predict(self, n_predictions): # pass context with k values
        """
        Generate predictions.
        :param n_predictions: Number of predictions to generate.
        """
        if len(self.history) < self.k:
            raise ValueError("Not enough data points to calculate the median for the specified window size (k).")
        last_k_values = self.history[-self.k:]
        median = np.median(last_k_values)
        return [median] * n_predictions

    def optimize(data, k_values):
        self.history = data
        best_k, best_loss = -1, np.inf
        for k in k_values:
            self.history = data[:k]
            for i in range(k, len(data)):
                last_k_values = self.history[-k:]
                prediction = np.median(last_k_values)
                target = data[k]
                loss += criterion(prediction, target)
            loss /= i
            if loss < best_loss:
                best_k = k
        return best_k

    def validate(self, val_data, config=None):
        """
        Predict and calculate validation loss.
        :param val_data: List or array of values.
        :param config: Placeholder for compatibility with similar models.
        """
        predictions = self.predict(len(val_data))
        targets = val_data

        loss = np.mean((np.array(predictions) - np.array(targets)) ** 2)
        return predictions, targets, loss

    def predict_next(self, new_value):
        """
        Update the model with new data and predict.
        :param new_value: The new observed value to add to the history.
        """
        self.history.append(new_value)
        if len(self.history) > self.k:
            self.history.pop(0)
        return np.median(self.history[-self.k:])

    def validate_forward(self, val_data):
        prediction = self.predict(1)
        predictions = [prediction]
        loss = 0
        for i in range(len(val_data)):
            loss += self.criterion(prediction, val_data[i])
            predictions.append(self.predict_next(val_data[i]))
        loss /= i
        targets = val_data # innecessary
        return predictions, targets, loss

    def validate_forward_dataloader(self, scaled_train_data, scaled_val_data): # do i even need to normalize for this?
        dataloader, original_length = get_dataloader(scaled_train_data, scaled_val_data, self.config, is_train=False, extra=True) # always false cause we are not going to validate the train set with SARIMA
        
        val_targets, val_predictions, extra_predictions = [], [], []
        n = 0
        # stride 1, step 1, batch size 1
        for inputs, target, scaled_inputs, scaled_targets, scaled_contexts, detrenders, transformers in dataloader: # this datalaoder returns inputs/contexts of size k
            # print(f"Scaled inputs shape {scaled_inputs.shape}")
            scaled_prediction = [np.median(scaled_inputs.cpu()[-self.k:])]
            # print(f"Scaled prediction {scaled_prediction}")
            common_scale_prediction = recover_original_prediction(scaled_prediction, scaled_contexts, detrenders, transformers, self.config['batch_size'], self.config['n_days']) 
            if n < original_length:
                val_predictions.append(common_scale_prediction)
                val_targets.append(target[0])
            else:
                extra_predictions.append(common_scale_prediction)
            n += 1

        loss = calculate_loss(torch.tensor(val_targets, dtype=torch.float32), torch.tensor(val_predictions, dtype=torch.float32), self.config)
        val_predictions.extend(extra_predictions)
        
        return np.array(val_predictions), np.array(val_targets), loss
            

def get_naive_model(config):
    return LastKMedian(config)
            