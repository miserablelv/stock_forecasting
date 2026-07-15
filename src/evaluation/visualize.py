from matplotlib import pyplot as plt
import numpy as np
import os

def visualize_monthly_data(data):
    plt.style.use("dark_background")
    
    plt.figure()
    data['MONTH'].plot()
    plt.title("Montly data before encoding")
    plt.savefig(fname="month_number.png", bbox_inches='tight')
    plt.show()

    plt.figure()
    data['Month_Sin'].plot()
    data['Month_Cos'].plot()
    plt.title("Montly data before encoding")
    plt.savefig(fname="month_number_enc.png", bbox_inches='tight')
    plt.show()

    return

def visualize_predictions(targets, predictions, loss, model_name, set): 
    base_path = f"{os.getcwd()}/best_models/{model_name}/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    preds_path = f"{base_path}{set}_predictions.png"
    plt.figure()
    plt.plot(targets, c='b', label=f'{set} targets')
    plt.plot(predictions, c='g', label=f'{set} predictions')
    plt.title(f"{set} data. Loss {loss}")
    plt.legend()
    plt.savefig(fname=preds_path, bbox_inches='tight')
    plt.show()
    return

def visualize_ranges(data, column_labels=None, title=""):
    plt.style.use("dark_background")
    
    column_mins = np.min(data, axis=0)
    column_maxs = np.max(data, axis=0)

    if column_labels is None:
        column_labels = [f'Col {i}' for i in range(data.shape[1])]

    plt.figure(figsize=(10, 6))

    max_deviation = column_maxs
    min_deviation = column_mins

    plt.bar(column_labels, max_deviation, color='skyblue', label='Max')#a desviación desde 0')
    plt.bar(column_labels, min_deviation, color='lightcoral', label='Min')# desviación desde 0')

    plt.axhline(0, color='black',linewidth=1)
    plt.title('Features distribution')
    plt.xlabel('Variables')
    plt.ylabel('Deviation from 0')
    plt.xticks(rotation=45, ha='right')

    plt.legend()

    plt.tight_layout()
    plt.savefig(fname=title, bbox_inches='tight')
    plt.show()


def display_strategies_statistics(strategies_results_dicts):
    """
    Display a comparison plot for the profit of each strategy in the dictionary list.
    
    Parameters:
    strategies_results_dicts (list): A list of dictionaries, each containing the keys:
                                     - 'profit': The profit value of the strategy.
                                     - 'model_name': The name of the model associated with the strategy.
    """
    # Extract strategy names, model names, and profits
    strategy_names = [f"Strategy {i+1}" for i in range(len(strategies_results_dicts))]
    strategy_profits = [strategy['profit'] if strategy is not None else 0 for strategy in strategies_results_dicts]
    model_names = [strategy['model_name'] if strategy is not None else "N/A" for strategy in strategies_results_dicts]
    
    print(f"Best strategies for models, profit: {strategy_profits}")
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(strategy_names, strategy_profits, color='skyblue', edgecolor='black')
    
    # Add titles and labels
    plt.title("Strategy Profit Comparison", fontsize=16)
    plt.xlabel("Strategies", fontsize=14)
    plt.ylabel("Profit", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Annotate each bar with the profit value and the model name
    for i, (bar, profit, model_name) in enumerate(zip(bars, strategy_profits, model_names)):
        # Profit annotation above the bar
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(strategy_profits), 
                 f"{profit:.2f}", ha='center', va='bottom', fontsize=10, color='black')
        
        # Model name annotation below the bar
        plt.text(bar.get_x() + bar.get_width() / 2, -0.05 * max(strategy_profits), 
                 model_name, ha='center', va='top', fontsize=10, color='gray', rotation=45)
    
    # Adjust plot layout to make room for annotations
    plt.tight_layout()
    plt.show()


# from collections import defaultdict

# def is_defaultdict_list_empty(d):
#     return isinstance(d, defaultdict) and d.default_factory is list and len(d) == 0


def visualize_models_losses(models_dict):    
    # Collect metrics for each model
    model_names = []
    avg_losses = []
    best_losses = []
    worst_losses = []
    std_devs = []
    
    # Calculate metrics for each model
    for model_name, losses in models_dict.items():
        model_names.append(model_name)
        losses_array = np.array(losses)
        
        avg_losses.append(np.mean(losses_array))
        best_losses.append(np.min(losses_array))
        worst_losses.append(np.max(losses_array))
        std_devs.append(np.std(losses_array))

    # Plotting
    x = np.arange(len(model_names))  # the label locations
    
    # Set up figure and axis for bar charts
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2  # the width of the bars
    
    # Plot bars for each metric
    ax.bar(x - width, avg_losses, width, label='Average Loss', color='skyblue')
    ax.bar(x, best_losses, width, label='Best Loss', color='limegreen')
    ax.bar(x + width, worst_losses, width, label='Worst Loss', color='salmon')
    ax.bar(x + 2*width, std_devs, width, label='Std Dev of Losses', color='orange')
    
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Loss')
    ax.set_title('Model Loss Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    
    # Show plot
    plt.tight_layout()
    path = "C:/Users/Rober/Downloads/stock_predictor/images/models_losses_comparison.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(fname=path, bbox_inches='tight', dpi=300)
    plt.show()

def visualize_ranked_losses(models_dict):
    # Helper function to plot ranked bar charts for a specific metric
    def plot_metric(metric_name, metric_values, color, ylabel, save_path):
        # Sort models by the current metric
        sorted_indices = np.argsort(metric_values)
        sorted_model_names = [model_names[i] for i in sorted_indices]
        sorted_metric_values = [metric_values[i] for i in sorted_indices]
        
        # Plot ranked bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_model_names, sorted_metric_values, color=color)
        plt.xlabel('Model')
        plt.ylabel(ylabel)
        plt.title(f'{metric_name} (Ranked)')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # Collect metrics for each model
    model_names = []
    avg_losses = []
    best_losses = []
    worst_losses = []
    std_devs = []
    
    for model_name, losses in models_dict.items():
        model_names.append(model_name)
        losses_array = np.array(losses)
        avg_losses.append(np.mean(losses_array))
        best_losses.append(np.min(losses_array))
        worst_losses.append(np.max(losses_array))
        std_devs.append(np.std(losses_array))
    
    # Plot ranked bar charts for each metric
    metrics = {
        'Average Loss': (avg_losses, 'skyblue', 'Loss'),
        'Best Loss': (best_losses, 'limegreen', 'Loss'),
        'Worst Loss': (worst_losses, 'salmon', 'Loss'),
        'Std Dev of Losses': (std_devs, 'orange', 'Standard Deviation')
    }
    
    for metric_name, (values, color, ylabel) in metrics.items():
        save_path = f"C:/Users/Rober/Downloads/stock_predictor/images/{metric_name.lower().replace(' ', '_')}_ranked.png"
        plot_metric(metric_name, values, color, ylabel, save_path)




import matplotlib.pyplot as plt
import numpy as np

def visualize_strategy_results(profit, win_rate, profit_factor, sortino_ratio): # MAYBE BETTER TO COMPARE EACH METRIC FOR EACH MODEL
    """
    Visualizes the results of a trading strategy.
    
    Parameters:
    - profit (float): Total profit or net return.
    - win_rate (float): Percentage of trades that were profitable.
    - profit_factor (float): Ratio of gross profit to gross loss.
    - sortino_ratio (float): Risk-adjusted return metric.
    """
    # Define metric names and values
    metrics = ['Profit ($)', 'Win Rate (%)', 'Profit Factor', 'Sortino Ratio']
    values = [profit, win_rate * 100, profit_factor, sortino_ratio]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar colors based on metric performance (adjust thresholds as needed)
    colors = [
        'green' if profit > 0 else 'red',                     # Profit
        'green' if win_rate >= 0.5 else 'red',                # Win rate
        'green' if profit_factor >= 1.5 else 'red',           # Profit factor
        'green' if sortino_ratio >= 1 else 'red'              # Sortino ratio
    ]
    
    # Plot bars
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Annotate each bar with its value
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.05, f"{value:.2f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    
    # Title and axis labels
    ax.set_title('Strategy Performance Summary', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xlabel('Metrics', fontsize=12)
    
    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    
    # Add a conclusion text box
    if profit > 0 and win_rate >= 0.5 and profit_factor >= 1.5 and sortino_ratio >= 1:
        conclusion_text = "The strategy performed well overall."
        conclusion_color = 'green'
    else:
        conclusion_text = "The strategy's performance has room for improvement."
        conclusion_color = 'red'
    
    # Add conclusion text below the chart
    ax.text(
        0.5, -0.2, conclusion_text, color=conclusion_color, fontsize=14, fontweight='bold',
        ha='center', va='center', transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor=conclusion_color)
    )
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


import os # later remove it and pass the path as param

def compare_strategies_evolutions(results_dfs, names, set="val_3", taxes=True):
    if taxes is True:
        to_add = "(Spanish taxes)"
    else:
        to_add = "(no taxes)"
    base_path = "C:/Users/Rober/Downloads/stock_predictor/images/"
    # for stat in results_dfs[0].columns:
    plt.figure()
    for strategy, name in zip(results_dfs, names):
        strategy['Equity'].plot(label=name)
    plt.title(f"Equity comparison on {set} set {to_add}")
    plt.xlabel("Date")
    plt.ylabel('Equity')
    plt.legend()
    plt.savefig(fname=base_path+f"equity_comparison_{"_".join(names)}_{set}_{to_add}.png", bbox_inches='tight')
    plt.show()
    return
            

def analyze_strategy_results(results_df):
    # plt.plot(results_df, labels='columns')
    # results_df.plot()
    results_df[['Cum_gained', 'Cum_lost']].plot()
    path = "C:/Users/Rober/Downloads/stock_predictor/images/strategy_results_new.png"
    plt.savefig(fname=path, bbox_inches='tight')
    plt.show()
    # of course i'll have to do different figures for each related range
    return

def compare_strategies(strategies_results):
    print(f"Comparing equity")
    for strategy in strategies_results:
        plt.figure()
        strategy['Equity'].plot()
        plt.show()
    print(f"Comparing cumulative gain and loss")
    for strategy in strategies_results:
        plt.figure()
        strategy['Cum_gained'].plot()
        strategy['Cum_lost'].plot()
        plt.show()
    # print(f"Comparing profit factor")
    # for strategy in strategies_results:
        # plt.plot(strategy['Equity'])
        # plt.show()
    print(f"Comparing WR evolution")
    for strategy in strategies_results:
        plt.plot(strategy['Wins']/(strategy['Losses']+0.001))
        plt.show()
    # print(f"Comparing equity")
    # for strategy in strategies_results:
    #     plt.plot(strategy['Equity'])
    #     plt.show()
    # print(f"Comparing equity")
    # for strategy in strategies_results:
    #     plt.plot(strategy['Equity'])
    #     plt.show()
    return


def compare_model_strategies(strategies_dict): # sell at the end to make it equal
    """ strategies_dict is a dictionary where each key is the model name
    and each value the strategy instance.
    each strategy instance contains data like strategy.n_wins, strategy.n_losses,
    strategy.initial_equity, strategy.final_equity"""

    # plot profit comparison

    # plot win rate comparison

    # plot maximum drawdown comparison

    # plot sortino ratio comparison


    active_strategies = {'Predictions': predictions_strategy}
                  # 'Predictions+Indicators': predictions_plus_indicators_strategy}

    strategies_profit = {'BuyAndHold': round(((reference_strategy.getBroker().getEquity()-10000)/10000)*100, 2),
                         'Predictions': round(((predictions_strategy.getBroker().getEquity()-10000)/10000)*100, 2)}
                         # 'Predictions+Indicators': ((predictions_plus_indicators_strategy.getBroker().getEquity()-10000)/10000)*100}

    plot_profit_comparison(strategies_profit)

    strategies_wr = {'BuyAndHold': calculate_wr(reference_strategy),
                     'Predictions': calculate_wr(predictions_strategy)}
                     # 'Predictions+Indicators': calculate_wr(predictions_plus_indicators_strategy)}
    
    plot_winrate_comparison(strategies_wr)
    
    relative_profits = {strategy: strategies_profit[strategy] - strategies_profit['BuyAndHold'] for strategy in active_strategies.keys()}

    w_strategy_name = max(relative_profits, key=relative_profits.get)
    w_strategy = active_strategies[w_strategy_name]
    best_profit = strategies_profit[w_strategy_name]
    relative_profit = relative_profits[w_strategy_name]

    print(f"Max profit: {best_profit}%, encountered in strategy {w_strategy}. Relative profit: {relative_profit}%")

# def analyze_best_config():
#     best_config_path = f"{os.getcwd()}best_overall_config" # could save it as a global variable
    # load test strategy instance