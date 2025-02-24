import pandas as pd


def calculate_profit_factor(trades):
    """
    Calculate the Profit Factor.
    :param trades: List of trade returns (positive for profits, negative for losses).
    :return: Profit Factor.
    """
    print(f"Trades {trades[0] / 2}")
    
    gross_profit = sum(x for x in trades if x > 0)
    gross_loss = abs(sum(x for x in trades if x < 0))

    if gross_loss == 0:  # Avoid division by zero
        return float('inf') if gross_profit > 0 else 0

    return gross_profit / gross_loss


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the Maximum Drawdown (MDD).
    :param cumulative_returns: List of cumulative portfolio values or returns over time.
    :return: Maximum Drawdown.
    """
    peak = cumulative_returns[0]
    max_drawdown = 0

    for value in cumulative_returns:
        peak = max(peak, value)  # Update the peak
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

import numpy as np

def calculate_sortino_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sortino Ratio.
    :param returns: List of periodic returns.
    :param risk_free_rate: Risk-free rate (default is 0).
    :return: Sortino Ratio.
    """
    excess_returns = [r - risk_free_rate for r in returns]
    downside_returns = [r for r in excess_returns if r < 0]

    downside_deviation = np.std(downside_returns, ddof=1)
    avg_excess_return = np.mean(excess_returns)

    if downside_deviation == 0:  # Avoid division by zero
        return float('inf') if avg_excess_return > 0 else 0

    return avg_excess_return / downside_deviation


def calculate_sharpe_ratio(strategy_results):
    equity_series = strategy_results['Equity']
    with pd.option_context("future.no_silent_downcasting", True):
        returns = equity_series.pct_change().dropna().infer_objects(copy=False)

    # Calculate Sharpe Ratio
    mean_return = returns.mean()
    std_dev_return = returns.std()
    sharpe_ratio = mean_return / (std_dev_return + 1e-8) + 1e-8 # Adding small epsilon to avoid divide-by-zero

    return sharpe_ratio

def calculate_sortino_ratio(strategy_results):
    equity_series = strategy_results['Equity']
    returns = equity_series.pct_change().dropna().infer_objects(copy=False)

    # Define a minimum acceptable return (MAR), usually 0 or risk-free rate
    mar = 0  # Replace with the risk-free rate if needed
    
    # Calculate downside deviation (standard deviation of negative returns relative to MAR)
    downside_returns = returns[returns < mar] - mar
    downside_deviation = np.sqrt((downside_returns**2).mean())
    
    # Calculate Sortino Ratio
    sortino_ratio = mean_return / (downside_deviation + 1e-8)  # Add epsilon to avoid divide-by-zero

    return sortino_ratio


def calculate_wr(strategy):
    try:
        wr = (strategy.wins / (strategy.wins + strategy.losses)) * 100
    except ZeroDivisionError:
        wr = 0
    return wr