from matplotlib import pyplot as plt
import numpy as np

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
    print(f"Loss: {loss}")
    
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

    plt.bar(column_labels, column_maxs, color='skyblue', label='Max')
    plt.bar(column_labels, column_mins, color='lightcoral', label='Min')

    plt.axhline(0, color='black',linewidth=1)
    plt.title('Features distribution')
    plt.xlabel('Variables')
    plt.ylabel('Deviation from 0')
    plt.xticks(rotation=45, ha='right')

    plt.legend()

    plt.tight_layout()
    plt.savefig(fname=title, bbox_inches='tight')
    plt.show()


def visualize_models_losses(models_dict):    
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

    x = np.arange(len(model_names)) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    
    ax.bar(x - width, avg_losses, width, label='Average Loss', color='skyblue')
    ax.bar(x, best_losses, width, label='Best Loss', color='limegreen')
    ax.bar(x + width, worst_losses, width, label='Worst Loss', color='salmon')
    ax.bar(x + 2*width, std_devs, width, label='Std Dev of Losses', color='orange')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Loss')
    ax.set_title('Model Loss Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    path = "C:/Users/Rober/Downloads/stock_predictor/images/models_losses_comparison.png"
    plt.savefig(fname=path, bbox_inches='tight')
    plt.show()



import os

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
            

def compare_strategies(strategies_results):
    print("Comparing equity")
    for strategy in strategies_results:
        plt.figure()
        strategy['Equity'].plot()
        plt.show()
    print("Comparing cumulative gain and loss")
    for strategy in strategies_results:
        plt.figure()
        strategy['Cum_gained'].plot()
        strategy['Cum_lost'].plot()
        plt.show()
    print("Comparing WR evolution")
    for strategy in strategies_results:
        plt.plot(strategy['Wins']/(strategy['Losses']+0.001))
        plt.show()
    return





def plot_profit_comparison(strategies_profit):
    print("Profit with different strategies", strategies_profit.items())
    
    plt.figure()
    plt.bar(x=np.arange(0, len(strategies_profit.keys())), 
            height=strategies_profit.values(), 
            align="center")
    plt.xticks(ticks=np.arange(0, len(strategies_profit.keys())), 
               labels=strategies_profit.keys(), 
               rotation=45, 
               ha="right")
    
    plt.xlabel('Strategies')
    plt.ylabel('Profit')
    plt.title('Profit by strategy (%)')
    
    plt.grid(axis='y')
    
    plt.show()
    
    
def plot_winrate_comparison(strategies_wr):
    plt.bar(x=np.arange(0, len(strategies_wr.keys())), 
            height=strategies_wr.values(), 
            align="center")
    plt.xticks(ticks=np.arange(0, len(strategies_wr.keys())), 
               labels=strategies_wr.keys(), 
               rotation=45, 
               ha="right")
    
    plt.xlabel('Strategies')
    plt.ylabel('Win rate')
    plt.title('WR by strategy (%)')
    
    plt.grid(axis='y')
    
    plt.show()

def calculate_wr(strategy):
    try:
        wr = (strategy.wins / (strategy.wins + strategy.losses)) * 100
    except ZeroDivisionError:
        wr = 0
    return wr