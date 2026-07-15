from hyperopt import hp

strategy_params = {
    'use_short': hp.choice('use_short', [True, False]),
    'scale': hp.choice('scale', ['og', 'com', 'double']),
    # 'substract_taxes': hp.choice('substract_taxes', [True, False]) # nonsense for optimization. no taxes will always provide better results
}

model_params = {
    'differential': hp.uniform('differential', 0.005, 0.07),
    'threshold_type': hp.choice('threshold_type', ['FIXED', 'VARIABLE']),
    'fixed_time': hp.choice('fixed_time', [True, False]),
    'use_indicators': hp.choice('use_indicators', [True, False]),
    'stride': 4,
    'atr_multiplier': hp.uniform('atr_multiplier', 0.6, 2.5)
}

indicators_params = {
    'rsi_time_period': hp.choice('rsi_time_period', [5, 7, 10, 14]),
    'macd_fast_period': hp.choice('macd_fast_period', [8, 10, 12]),
    'macd_slow_period': hp.choice('macd_slow_period', [20, 22, 26]),
    'signal_period': hp.choice('signal_period', [5, 7, 9]),
    'atr_period': hp.choice('atr_period', [10, 14, 20]),
    'obv_trend_period': hp.choice('obv_trend_period', [5, 10, 15, 20, 30]),
    'obv_trend_multiplier': hp.choice('obv_trend_multiplier', [1, 1.2, 1.4, 1.6, 1.8, 2])
}
