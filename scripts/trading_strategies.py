from pyalgotrade import strategy
from pyalgotrade.technical import ma, rsi, macd

import numpy as np

from data import *

from models.models_utils import get_initial_context

from time import sleep

from sys import exit

import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

class PredictionBasedStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, predictions, data, strategy_params, indicators):
        super(PredictionBasedStrategy, self).__init__(feed, 100_000)
        self.initial_investment = 100_000
        self.feed = self.getFeed()
        self.instrument = instrument
        self.initial_equity = 100000

        # for real time predictions
        self.previous_data = None
        # self.test_data = iter(test_data) # for now just precalculated
        # self.model = model
        self.threshold_type = strategy_params['threshold_type']
        self.current_threshold = strategy_params['differential']
        self.use = strategy_params['use']
        self.use_short = strategy_params['use_short']
        self.stride = strategy_params['stride']
        self.atr_multiplier = strategy_params['atr_multiplier']
        self.fixed_time = strategy_params['fixed_time']
        self.taxes = strategy_params['substract_taxes']

        self.predictions = predictions
        self.data = data
        self.indicator_params = strategy_params['indicators']
        
        self.setUseAdjustedValues(True)
        self.__position = None
        self.last_year = None
        self.yearly_realized_returns = 0
        self.n_shares = 0
        self.position_type = None
        self.days_until_prediction = 0
        
        self.wins = 0
        self.losses = 0
        self.cum_gained = 0
        self.cum_lost = 0

        
        self.sign_idx = 0

        self.trades = []
        self.cum_returns = [self.initial_investment]

        self.results = pd.DataFrame(index=predictions.index, columns=['Equity', 'Cash', 'Cum_gained', 'Cum_lost', 'Wins', 'Losses']) # i will take out cash because it doesn't work very well
        
        # self.getBroker().setCommission(0.001)
        adj_close = np.array(feed[instrument].getAdjCloseDataSeries())

        self.rsi, self.macd, self.macd_signal, self.obv, self.atr = indicators
        

    def calculate_stop_loss(self, share_price):
        atr_value = self.atr[-1]
        return share_price - self.atr_multiplier * atr_value

    def calculate_current_threshold(self):
        # calculating the current threshold for predicted change as a multiple of the current ATR
        self.current_threshold = self.atr_multiplier * (self.atr.loc[self.getCurrentDateTime()].values[0] / 10000) # 100x100 (to percentage, to proportion)
    
        return self.current_threshold

    def check_entry(self, bar):
        """
        Reversal factors:
        1. RSI crosses < 30 or > 70
        2. MACD crosses line
        3. Predictions clear direction
        """
        rsi_value = self.rsi.loc[self.getCurrentDateTime()].values[0]
        rsi_value_yesterday = self.rsi.loc[self.getCurrentDateTime()-datetime.timedelta(days=7)].values[0]
        macd_value = self.macd.loc[self.getCurrentDateTime()].values[0]
        macd_value_yesterday = self.macd.loc[self.getCurrentDateTime()-datetime.timedelta(days=7)].values[0]
        macd_signal_value = self.macd_signal.loc[self.getCurrentDateTime()].values[0]
        macd_signal_value_yesterday = self.macd_signal.loc[self.getCurrentDateTime()-datetime.timedelta(days=7)].values[0]
        obv_value = self.obv.loc[self.getCurrentDateTime()].values[0]

        volume_spike_reversal = bar.getVolume() > self.indicator_params['obv_trend_multiplier'] * talib.SMA(np.array(self.feed[self.instrument].getVolumeDataSeries()), timeperiod=self.indicator_params['obv_trend_period'])[-1] ## can precalcualte it like the others. NOT using it currently

        rsi_bullish_reversal = rsi_value > 35 and rsi_value_yesterday < 35
        rsi_bearish_reversal = rsi_value < 70 and rsi_value_yesterday > 70

        macd_bullish_reversal = macd_value > macd_signal_value and macd_value_yesterday < macd_signal_value_yesterday # cross down
        macd_bearish_reversal = macd_value < macd_signal_value and macd_value_yesterday > macd_signal_value_yesterday # cross up


        prediction_bullish_reversal = (self.current_predicted_change > self.current_threshold) and not (self.current_predicted_change < -self.current_threshold)
        prediction_bearish_reversal = (self.current_predicted_change < -self.current_threshold) and not (self.current_predicted_change > self.current_threshold)

        enter_long = False
        enter_short = False
        
        if self.use == "PREDICTIONS+INDICATORS":
            if (rsi_bullish_reversal + macd_bullish_reversal) > 0 or prediction_bullish_reversal:
                enter_long = True
            
            if (rsi_bearish_reversal + macd_bearish_reversal) > 0 and prediction_bearish_reversal:
                enter_short = True
                
        elif self.use == "PREDICTIONS":
            if prediction_bullish_reversal:
                enter_long = True
            if prediction_bearish_reversal:
                enter_short = True    

        else:
            raise Error("wrong use")

        if (enter_long and not enter_short) or (enter_long and not self.use_short):
            return "ENTER LONG"
        if enter_short and not enter_long and self.use_short:
            return "ENTER SHORT"
        else:
            return "WAIT"

    def check_continuation(self):
        """
        Continuation factors:
        1. MACD increases in absolute value
        2. Predictions keep clearly pointing in the same direction
        3. OBV increasing/decreasing above its moving average
        """
        overbought = self.rsi.loc[self.getCurrentDateTime()].values[0] > 70
        previous_date = self.getCurrentDateTime()-datetime.timedelta(days=7)
        macd_increasing = abs(self.macd.loc[self.getCurrentDateTime()].values[0]) > abs(self.macd.loc[previous_date].values[0])
        obv_trending = abs(self.obv.loc[self.getCurrentDateTime()].values[0]) > abs(talib.SMA(self.obv.loc[:self.getCurrentDateTime()].values.flatten(), timeperiod=self.indicator_params['obv_trend_period'])[-1]) ## might precalculate it like the others
        if self.position_type == "LONG":
            preds_continuation = (self.current_predicted_change > self.current_threshold / 2) and not (self.current_predicted_change < self.current_threshold / 2)
        else:
            preds_continuation = (self.current_predicted_change < self.current_threshold) and not (self.current_predicted_change > self.current_threshold)

        if self.use == "PREDICTIONS+INDICATORS":
            if (macd_increasing + obv_trending + preds_continuation) > 0 and not overbought:
                return "HOLD"
            else:
                return "EXIT"
        elif self.use == "PREDICTIONS":
            if preds_continuation:
                return "HOLD"
            else:
                return "EXIT"
        else:
            raise Error("wrong use")
        

    def calculate_take_profit(self, share_price): # not used yet
        atr_value = self.atr.loc[self.getCurrentDateTime()]
        return share_price + self.atr_multiplier * atr_value
    
    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        # print("Position opened:", execInfo.getDateTime())

    def enterPosition(self):
        if self.__position is not None:
            self.__position = None
        if self.position_type == "LONG":
            self.__position = self.enterLong(self.instrument, self.n_shares, goodTillCanceled=False, allOrNone=True)
        elif self.position_type == "SHORT":
            self.__position = self.enterShort(self.instrument, self.n_shares, goodTillCanceled=False, allOrNone=True)
        else:
            raise ValueError(f"Unrecognized position type: {self.position_type}")

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        # print("Position closed:", execInfo.getDateTime())
        position_pnl = position.getPnL()
        if position_pnl > 0.0:
            self.wins += 1
            self.cum_gained += position_pnl
        else:
            self.losses += 1
            self.cum_lost += position_pnl
            
        self.yearly_realized_returns += position_pnl
        self.cum_returns.append(self.cum_returns[-1] + position_pnl)
        self.trades.append(position_pnl)
        self.__position = None

    def onCancelOk(self, position):
        self.__position = None

    def onEnterCanceled(self, position):
        self.__position = None


    def calculate_tax_deduction(self): # taxes in Spain
        if self.yearly_realized_returns < 0:
            deduction = 0
        elif 0 < self.yearly_realized_returns < 6000:
            deduction = self.yearly_realized_returns*0.19
        elif 6000 <= self.yearly_realized_returns < 50000:
            deduction = self.yearly_realized_returns*0.21
        elif 50000 <= self.yearly_realized_returns < 200000:
            deduction = self.yearly_realized_returns*0.23
        else:
            deduction = self.yearly_realized_returns*0.26
        print(f'{deduction}$ taxed on year', self.last_year)

        return deduction

    def real_time_prediction(self, new_data):
        print(f"Creating prediction for week number {self.i}")
        self.context = np.concatenate((self.context[1:], new_data))
        print(f"Context shape in real time prediction {self.context.shape}")
        self.hidden, scaled_context, scaled_prediction, scaler, detrender = self.model.get_prediction(self.context, self.hidden)
        # maybe better try to receive directly the original prediction.
        prediction = recover_original_prediction(scaled_prediction, scaled_context, scaler, detrender, self.stride)
        
        return prediction
        
    def check_taxes(self, bars):
        current_datetime = bars.getBar(self.instrument).getDateTime()
        current_year = current_datetime.year

        if self.last_year is None:
            self.last_year = current_year

        if current_year != self.last_year:
            print(f"Yearly realized gain for {self.last_year}: {self.yearly_realized_returns}")

            deduction = self.calculate_tax_deduction()

            self.getBroker().setCash(self.getBroker().getCash()-deduction)
            self.last_year = current_year
            self.yearly_realized_returns = 0


    def update_results(self):
        self.results.loc[self.getCurrentDateTime()] = [self.getBroker().getEquity(), self.getBroker().getCash(), self.cum_gained, self.cum_lost, self.wins, self.losses]
        # could save tax stats too

    def onBars(self, bars):        
        self.update_results() # maybe at the end better?
        if self.threshold_type == "VARIABLE":
            self.current_threshold = self.atr_multiplier * (self.atr.loc[self.getCurrentDateTime()].values[0] / 10000)
            
        bar = bars[self.instrument]

        
        share_price = bar.getClose()

        predictions = self.predictions.loc[self.getCurrentDateTime()].values
        current_price = self.data.loc[self.getCurrentDateTime()].values

        pct_difs = np.abs(predictions[1:]/current_price - 1)
        self.current_predicted_change = pct_difs[np.argmax(pct_difs)]
        steps_until_pred = np.argmax(pct_difs) + 1

        if self.__position is not None:
            if self.fixed_time == True:
                if self.getCurrentDateTime() == self.close_position_date:
                    self.__position.exitMarket()
            else:
                action = self.check_continuation()
                if action == "HOLD":
                    return
                elif action == "EXIT":
                    self.__position.exitMarket()
                else:
                    raise ValueError(f"Unrecognized action: {action}")
        else: # no position opened
            self.n_shares = (self.getBroker().getCash() / bar.getAdjClose()) * 0.99                
            if self.n_shares > 0: # can at least buy one share
                action = self.check_entry(bar)
                if action == "ENTER LONG":
                    self.position_type = "LONG"
                    self.enterPosition()
                    self.close_position_date = self.getCurrentDateTime()+datetime.timedelta(days=int(steps_until_pred*7))
                elif action == "ENTER SHORT":
                    self.position_type = "SHORT"
                    self.enterPosition()
                    self.close_position_date = self.getCurrentDateTime()+datetime.timedelta(days=int(steps_until_pred*7))
                # else: # don't do anything
                    # continue
            else:
                print("Insufficient cash to enter position. Cash:", cash, "share price", share_price)

        if self.taxes is True:
            self.check_taxes(bars)


class IndicatorsBasedStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, data, strategy_params, indicators):
        super(IndicatorsBasedStrategy, self).__init__(feed, 100_000)
        self.initial_investment = 100_000
        self.feed = self.getFeed()
        self.instrument = instrument

        self.use_short = strategy_params['use_short']
        self.atr_multiplier = strategy_params['atr_multiplier']
        self.fixed_time = strategy_params['fixed_time']
        self.taxes = strategy_params['substract_taxes']
    
        self.data = data
        self.indicator_params = strategy_params['indicators']
        
        self.setUseAdjustedValues(True)
        self.__position = None
        self.last_year = None
        self.yearly_realized_returns = 0
        self.n_shares = 0
        self.position_type = None
        self.days_until_prediction = 0

        self.trades = []
        self.wins = 0
        self.losses = 0
        self.cum_gained = 0
        self.cum_lost = 0

        self.cum_returns = [self.initial_investment]

        self.results = pd.DataFrame(index=data.index, columns=['Equity', 'Cash', 'Cum_gained', 'Cum_lost', 'Wins', 'Losses']) # i will take out cash because it doesn't work very well
        
        # self.getBroker().setCommission(0.001)

        # Technical indicators
        self.rsi, self.macd, self.macd_signal, self.obv, self.atr = indicators

    def calculate_stop_loss(self, share_price):
        atr_value = self.atr[-1]
        return share_price - self.atr_multiplier * atr_value

    def check_entry(self, bar):
        rsi_value = self.rsi.loc[self.getCurrentDateTime()].values[0]
        rsi_value_yesterday = self.rsi.loc[self.getCurrentDateTime()-datetime.timedelta(days=7)].values[0]
        macd_value = self.macd.loc[self.getCurrentDateTime()].values[0]
        macd_value_yesterday = self.macd.loc[self.getCurrentDateTime()-datetime.timedelta(days=7)].values[0]
        macd_signal_value = self.macd_signal.loc[self.getCurrentDateTime()].values[0]
        macd_signal_value_yesterday = self.macd_signal.loc[self.getCurrentDateTime()-datetime.timedelta(days=7)].values[0]
        obv_value = self.obv.loc[self.getCurrentDateTime()].values[0]

        volume_spike_reversal = bar.getVolume() > self.indicator_params['obv_trend_multiplier'] * talib.SMA(np.array(self.feed[self.instrument].getVolumeDataSeries()), timeperiod=self.indicator_params['obv_trend_period'])[-1] ## can precalculate it like the others

        rsi_bullish_reversal = rsi_value > 35 and rsi_value_yesterday < 35
        rsi_bearish_reversal = rsi_value < 70 and rsi_value_yesterday > 70

        macd_bullish_reversal = macd_value > macd_signal_value and macd_value_yesterday < macd_signal_value_yesterday # cross down
        macd_bearish_reversal = macd_value < macd_signal_value and macd_value_yesterday > macd_signal_value_yesterday # cross up

        enter_long = False
        enter_short = False
    
        if (rsi_bullish_reversal + macd_bullish_reversal) > 0 and volume_spike_reversal:
            enter_long = True
        
        if (rsi_bearish_reversal + macd_bearish_reversal) > 1:
            enter_short = True

        if (enter_long and not enter_short) or (enter_long and not self.use_short):
            return "ENTER LONG"
        if enter_short and not enter_long and self.use_short:
            return "ENTER SHORT"
        else:
            return "WAIT"

    def check_continuation(self):
        overbought = self.rsi.loc[self.getCurrentDateTime()].values[0] > 70
        previous_date = self.getCurrentDateTime()-datetime.timedelta(days=7)
        macd_increasing = abs(self.macd.loc[self.getCurrentDateTime()].values[0]) > abs(self.macd.loc[previous_date].values[0])
        obv_trending = abs(self.obv.loc[self.getCurrentDateTime()].values[0]) > abs(talib.SMA(self.obv.loc[:self.getCurrentDateTime()].values.flatten(), timeperiod=self.indicator_params['obv_trend_period'])[-1]) ## can precalcualte it like the others

        if macd_increasing or obv_trending:
            return "HOLD"
        else:
            return "EXIT"
        

    def calculate_take_profit(self, share_price): # not used yet
        atr_value = self.atr.loc[self.getCurrentDateTime()]
        return share_price + self.atr_multiplier * atr_value
    
    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        # print("Position opened:", execInfo.getDateTime())

    def enterPosition(self):
        if self.__position is not None:
            self.__position = None
        if self.position_type == "LONG":
            self.__position = self.enterLong(self.instrument, self.n_shares, goodTillCanceled=False, allOrNone=True)
        elif self.position_type == "SHORT":
            self.__position = self.enterShort(self.instrument, self.n_shares, goodTillCanceled=False, allOrNone=True)
        else:
            raise ValueError(f"Unrecognized position type: {self.position_type}")

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        # print("Position closed:", execInfo.getDateTime())
        position_pnl = position.getPnL()
        if position_pnl > 0.0:
            self.wins += 1
            self.cum_gained += position_pnl
        else:
            self.losses += 1
            self.cum_lost += position_pnl
            
        self.yearly_realized_returns += position_pnl
        self.cum_returns.append(self.cum_returns[-1] + position_pnl)
        self.trades.append(position_pnl)
        self.__position = None

    def onCancelOk(self, position):
        self.__position = None

    def onEnterCanceled(self, position):
        self.__position = None

    def calculate_tax_deduction(self): # basado en los impuestos de España ### USE THIS
        if self.yearly_realized_returns < 0:
            deduction = 0
        elif 0 < self.yearly_realized_returns < 6000:
            deduction = self.yearly_realized_returns*0.19
        elif 6000 <= self.yearly_realized_returns < 50000:
            deduction = self.yearly_realized_returns*0.21
        elif 50000 <= self.yearly_realized_returns < 200000:
            deduction = self.yearly_realized_returns*0.23
        else:
            deduction = self.yearly_realized_returns*0.26
        print(f'{deduction}$ taxed on year', self.last_year)

        return deduction
        
    def check_taxes(self, bars):
        current_datetime = bars.getBar(self.instrument).getDateTime()
        current_year = current_datetime.year

        if self.last_year is None:
            self.last_year = current_year

        if current_year != self.last_year:
            print(f"Yearly realized gain for {self.last_year}: {self.yearly_realized_returns}")

            deduction = self.calculate_tax_deduction()

            self.getBroker().setCash(self.getBroker().getCash()-deduction)
            self.last_year = current_year
            self.yearly_realized_returns = 0


    def update_results(self):
        self.results.loc[self.getCurrentDateTime()] = [self.getBroker().getEquity(), self.getBroker().getCash(), self.cum_gained, self.cum_lost, self.wins, self.losses]

    def onBars(self, bars):        
        self.update_results() # maybe at the end better?
            
        bar = bars[self.instrument]
        
        share_price = bar.getClose()

        current_price = self.data.loc[self.getCurrentDateTime()] # sometimes scaled, sometimes not

        if self.__position is not None:
                action = self.check_continuation()
                if action == "HOLD":
                    return
                elif action == "EXIT":
                    self.__position.exitMarket()
                else:
                    raise ValueError(f"Unrecognized action: {action}")
        else: # no position opened
            self.n_shares = (self.getBroker().getCash() / bar.getAdjClose()) * 0.99                
            if self.n_shares > 0: # can at least buy one share
                action = self.check_entry(bar)
                if action == "ENTER LONG":
                    self.position_type = "LONG"
                    self.enterPosition()
                elif action == "ENTER SHORT":
                    self.position_type = "SHORT"
                    self.enterPosition()
                # else: # don't do anything
                    # continue
            else:
                print("Insufficient cash to enter position. Cash:", cash, "share price", share_price)

        if self.taxes is True:
            self.check_taxes(bars)




class BuyAndHoldStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, index):
        super(BuyAndHoldStrategy, self).__init__(feed, 100_000)
        self.initial_investment=10_000
        self.__position = None
        self.feed = self.getFeed()
        # self.instrument = instrument
        self.__instrument = instrument
        self.setUseAdjustedValues(True)
        self.__n_shares = None
        self.counter = 0
        self.wins, self.losses = 0, 0
        self.results = pd.DataFrame(index=index, columns=['Equity', 'Cash', 'Cum_gained', 'Cum_lost', 'Wins', 'Losses'])

    def onEnterCanceled(self, position):
        self.__position = None
        self.__n_shares = None

    def update_results(self):
        self.results.loc[self.getCurrentDateTime()] = [self.getBroker().getEquity(), self.getBroker().getCash(), 0, 0, self.wins, self.losses]
        # print(f"Updating results on day... {self.getCurrentDateTime()}")

    def onBars(self, bars): # buy on first candle
        self.update_results()
        if self.__position is None:
            bar = bars[self.__instrument]
            cash = self.getBroker().getCash()
            if self.__n_shares is not None:
                self.__n_shares = min(self.__n_shares*0.99, self.getBroker().getCash()/bar.getAdjClose())
            else:
                self.__n_shares = (self.getBroker().getCash()/bar.getAdjClose()) * 0.98
            self.__position = self.enterLong(self.__instrument, np.floor(self.__n_shares*10)/10.0)

    # def onFinish(self, bars): # para hacerlo en igualdad de condiciones, podría cerrar al final y deducir los impuestos
        # realized_return = self.__position.getPnL()
        # self.__position.exitMarket()
        # self.getBroker().setCash(self.initial_investment + realized_return * 0.8)
        # if realized_return > 0:

# def calculate_compound_attributes(results_df):
    