#------------------------------------------------------------------------------
###            Import Packages
#------------------------------------------------------------------------------
import sys, os
sys.path.append('D:\\MYCOURSES\\SHREE_DWARKADHISH_V3')  # Add project root to path
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import nbformat
from datetime import date, timedelta, datetime
import concurrent.futures
import yfinance as yf
import pretty_errors

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from loguru import logger

from typing import List, Dict, Optional
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
###            Import Modules 
#------------------------------------------------------------------------------
from config import cfg_nifty, cfg_vars
from src.utils import util_funcs, util_bear, util_bull

logger.add(cfg_vars.scanner_log_path, 
           rotation="50 MB", 
           retention="10 days", 
           level="DEBUG", 
           backtrace=True,  # Include full traceback for exceptions
           diagnose=True)   # Include additional context about the error

#------------------------------------------------------------------------------


class SupertrendGenerator:
    """
    Generates trading signals based on EMA strategy.
    Separated from data fetching and backtesting logic.
    """
    
    def __init__(self, data: pd.DataFrame, symbol: str = "Unknown"):
        """
        Initialize the signal generator with OHLC data.
        
        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                  and DatetimeIndex
            symbol: Symbol name for reference
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.data = data.copy()
        # Convert required columns to numeric to avoid type errors
        for col in required_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.symbol = symbol
        self.signals = []
        
    #------------------------------------------------------------------------------
    #                      PRICE-ACTION INDICATORS
    #------------------------------------------------------------------------------
    
    def price_action(self) -> pd.DataFrame:
        """
        Generate the ema_38_channel_flag column in the data, delayed by 2 candles.
        EMA3  = fastest (impulse)
        EMA8  = structure
        EMA200 = regime
        Returns:
            Updated DataFrame with 'ema_38_channel_flag' column added.
        """

        """Calculate all 5 EMAs required for the strategy."""
        
        # Orange EMAs (Length = 3)
        self.data['EMA3_High'] = self.data['High'].ewm(span=3, adjust=False).mean()
        self.data['EMA3_Low'] = self.data['Low'].ewm(span=3, adjust=False).mean()
        
        # White EMAs (Length = 8)
        self.data['EMA8_High'] = self.data['High'].ewm(span=8, adjust=False).mean()
        self.data['EMA8_Low'] = self.data['Low'].ewm(span=8, adjust=False).mean()
        
        # Close EMAs for cross detection
        self.data['EMA3_Close'] = self.data['Close'].ewm(span=3, adjust=False).mean()
        self.data['EMA8_Close'] = self.data['Close'].ewm(span=8, adjust=False).mean()
        
        #Will use it later for calculating Support & Resistance Zones
        self.data['ema20'] = self.data['Close'].ewm(span=20, adjust=False).mean()
        self.data['ema50'] = self.data['Close'].ewm(span=50, adjust=False).mean()

        #Analyze the EMA crosses and set flags
        self.data['ema_38_close_flag'] = None
        self.data['ema_38_channel_flag'] = None
        
        for i in range(100, len(self.data) - 2):  # Ensure i+2 is within bounds
            entry_idx = i + 2

            # Check for rising cross at confirmation bar i
            if (self.data['EMA3_Low'].iloc[i] > self.data['EMA8_Low'].iloc[i] and
                self.data['EMA3_High'].iloc[i] > self.data['EMA8_High'].iloc[i]):
                self.data.at[self.data.index[entry_idx], 'ema_38_channel_flag'] = 'rising'
            # Check for falling cross at confirmation bar i
            elif (self.data['EMA3_High'].iloc[i] < self.data['EMA8_High'].iloc[i] and
                  self.data['EMA3_Low'].iloc[i] < self.data['EMA8_Low'].iloc[i]):
                self.data.at[self.data.index[entry_idx], 'ema_38_channel_flag'] = 'falling'
        
        # Set ema_38_close_flag based on EMA positions
        conditions = [
            (self.data['EMA3_Close'] < self.data['EMA8_Close']) & (self.data['EMA3_Close'] > self.data['EMA200']) & (self.data['EMA8_Close'] > self.data['EMA200']),
            (self.data['EMA3_Close'] > self.data['EMA8_Close']) & (self.data['EMA3_Close'] > self.data['EMA200']) & (self.data['EMA8_Close'] > self.data['EMA200']),
            (self.data['EMA3_Close'] < self.data['EMA8_Close']) & (self.data['EMA3_Close'] < self.data['EMA200']) & (self.data['EMA8_Close'] < self.data['EMA200']),
            (self.data['EMA3_Close'] > self.data['EMA8_Close']) & (self.data['EMA3_Close'] < self.data['EMA200']) & (self.data['EMA8_Close'] < self.data['EMA200'])
        ]
        choices = ['bull_pullback', 'rising', 'falling', 'bear_pullback']
        self.data['ema_38_close_flag'] = np.select(conditions, choices, default=None)

        #print(f"[{self.symbol}] Channel overlay generated")
        return self.data

    #------------------------------------------------------------------------------
    #                      TREND INDICATORS 
    #------------------------------------------------------------------------------
    def calculate_ema200(self) -> pd.DataFrame:
        """Calculate 200-day EMA and set ema200_trend_flag."""
        self.data['EMA200'] = self.data['Close'].ewm(span=200, adjust=False).mean()
        #self.data['ema200_slope'] = self.ema_slope(self.data['EMA200'], lookback=20)

        self.data['ema200_trend_flag'] = np.where(self.data['Close'] > self.data['EMA200'], 'rising', 'falling')
        # Set ema200_direction
        self.data['ema200_direction'] = np.where(
            self.data['EMA200'] > self.data['EMA200'].shift(10), 'rising',
            np.where(self.data['EMA200'] < self.data['EMA200'].shift(10), 'falling', 'flat')
        )
        return self.data  

    def compute_supertrend(self,df, atr_period=10, factor=3.0):
        """
        df must contain columns: ['high', 'low', 'close']
        Computes supertrend and direction, and sets related columns on self.data.
        """

        high = df['high']
        low = df['low']
        close = df['close']

        # ───────── ATR Calculation (Wilder) ─────────
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

        # ───────── BASIC BANDS ─────────
        hl2 = (high + low) / 2
        upper_basic = hl2 + factor * atr
        lower_basic = hl2 - factor * atr

        # ───────── FINAL BANDS ─────────
        upper_final = upper_basic.copy()
        lower_final = lower_basic.copy()

        for i in range(1, len(df)):
            upper_final.iloc[i] = (
                min(upper_basic.iloc[i], upper_final.iloc[i-1])
                if close.iloc[i-1] <= upper_final.iloc[i-1]
                else upper_basic.iloc[i]
            )

            lower_final.iloc[i] = (
                max(lower_basic.iloc[i], lower_final.iloc[i-1])
                if close.iloc[i-1] >= lower_final.iloc[i-1]
                else lower_basic.iloc[i]
            )

        # ───────── SUPERTREND & DIRECTION ─────────
        supertrend = pd.Series(index=df.index, dtype='float64')
        direction = pd.Series(index=df.index, dtype='int')

        direction.iloc[0] = 1  # default = downtrend
        supertrend.iloc[0] = np.nan

        for i in range(1, len(df)):
            if close.iloc[i] > upper_final.iloc[i-1]:
                direction.iloc[i] = -1   # uptrend
            elif close.iloc[i] < lower_final.iloc[i-1]:
                direction.iloc[i] = 1    # downtrend
            else:
                direction.iloc[i] = direction.iloc[i-1]

            supertrend.iloc[i] = (
                lower_final.iloc[i] if direction.iloc[i] == -1
                else upper_final.iloc[i]
            )
        self.data['direction'] = direction
        self.data['supertrend'] = np.ceil(supertrend)
        self.data['atr'] = atr
        #print(f"[{self.symbol}] Signals generated with supertrend, and direction")
    
    #------------------------------------------------------------------------------
    #                      MOMENTUM INDICATORS
    #------------------------------------------------------------------------------
    def calculate_rsi(self, period=14):
        """Calculate RSI using pandas-ta and set momentum_flag."""
        self.data['RSI'] = np.ceil(ta.rsi(self.data['Close'], length=period))
        # Added RSI slope and direction
        self.data['RSI_slope'] = self.data['RSI'] - self.data['RSI'].shift(1)
        self.data['RSI_direction'] = np.where(self.data['RSI_slope'] > 0, 'increasing', 
                                              np.where(self.data['RSI_slope'] < 0, 'decreasing', 'flat'))
        # Updated momentum_flag: rising if RSI > 50 and rising, falling if RSI < 50 and falling
        self.data['momentum_flag'] = np.where(
            (self.data['RSI_direction'] == 'increasing') & (self.data['RSI'] > 50), 'rising',
            np.where(
                (self.data['RSI_direction'] == 'decreasing') & (self.data['RSI'] < 50), 'falling', ''
            )
        )
        return self.data
    #------------------------------------------------------------------------------
    #                     VOLUME INDICATORS
    #------------------------------------------------------------------------------
    def calculate_rvol(self) -> pd.DataFrame:
        """Calculate relative volume and set volume_flag."""
        self.data['avg_volume_20'] = self.data['Volume'].rolling(window=20).mean()
        self.data['rvol'] = self.data['Volume'] / self.data['avg_volume_20']
        self.data['volume_flag'] = np.where(
            self.data['rvol'] > 1.50, 'rising',
            np.where(
                (self.data['rvol'] >= 1.00) & (self.data['rvol'] <= 1.50), 'normal', 'falling'
            )
        )
        return self.data
    
    #------------------------------------------------------------------------------
    #                    VOLATILITY INDICATORS
    #------------------------------------------------------------------------------
    def trade_allowed(self, df, bbw_len=20, er_len=10, bbw_thr=0.04, er_thr=0.25):
        close = df['close']

        ma = close.rolling(bbw_len).mean()
        sd = close.rolling(bbw_len).std()
        bbw = (2 * sd) / ma

        direction = (close - close.shift(er_len)).abs()
        volatility = close.diff().abs().rolling(er_len).sum()
        er = direction / volatility

        bbw_dynamic = bbw > bbw.rolling(50).mean()
        bbw_expanding = bbw > bbw.shift(1)

        tradeable = (
            bbw_dynamic &
            bbw_expanding &
            (er > er_thr)
        )

        return tradeable


    #------------------------------------------------------------------------------
    #                   ENTRY/EXIT SIGNAL LOGIC
    #------------------------------------------------------------------------------
    def true_candle_range(self, df):
        """
        Calculates True Range for each candle
        Returns a pandas Series
        """
        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range
        
    #------------------------------------------------------------------------------
    #                    IMPLEMENT EXIT CRITERIA
    #------------------------------------------------------------------------------
    def calculate_support_n_resistance(self):
        """Calculate support and resistance levels."""
        pass  # Placeholder for future implementation

        self.data['sup_s1_low'] = self.data['ema20'] - 0.5 * self.data['atr']
        self.data['sup_s1_high'] = self.data['ema20'] + 0.5 * self.data['atr']
        self.data['sup_deep'] = self.data['ema50'] - 0.5 * self.data['atr']
    

        self.data['res_r1'] = self.data['ema20'] + 1.0 * self.data['atr']
        self.data['res_r2'] = self.data['ema20'] + 2.0 * self.data['atr']
        self.data['res_r3'] = self.data['ema20'] + 3.0 * self.data['atr']
        return self.data

    #------------------------------------------------------------------------------
    #                    GENERATE SIGNALS FROM ALL INDICATORS
    #------------------------------------------------------------------------------
    def generate_signals(self, lookback_bars: int = 10) -> List[Dict]:
        """
        Generate buy and sell signals based on the strategy rules.
        
        Args:
            lookback_bars: Number of bars to look back for pullback detection
            
        Returns:
            List of signal dictionaries
        """
        
        self.signals = []
        for i in range(100, len(self.data)):  # Start after EMAs stabilize
            #print(f"Index {i}....")
            #Price Action Flags
            ema_38_close_flag = self.data['ema_38_close_flag'].iloc[i]
            ema_38_channel_flag = self.data['ema_38_channel_flag'].iloc[i]
            
            #Trend Flags
            ema200_trend_flag = self.data['ema200_trend_flag'].iloc[i]
            ema200_direction = self.data['ema200_direction'].iloc[i]
            trend = self.data['trend'].iloc[i]
            trend_change_direction = self.data['trend_change_direction'].iloc[i]
            #st_stop_loss = self.data['st_stop_loss'].iloc[i]
            #st_resistance = self.data['st_resistance'].iloc[i]
            supertrend = self.data['supertrend'].iloc[i]   

            #Volume Flags
            volume_flag = self.data['volume_flag'].iloc[i]

            #Momentum Flags
            momentum_flag = self.data['momentum_flag'].iloc[i]
            rsi = self.data['RSI'].iloc[i]

            #Volatility/Sideways Flags
            tradeable = self.data['tradeable'].iloc[i]

            #Entry Candle Range
            range_ok = self.data['range_ok'].iloc[i]

            #Support & Resistance Levels
            sup_s1_low = round(self.data['sup_s1_low'].iloc[i],2)
            sup_s1_high = round(self.data['sup_s1_high'].iloc[i],2) 
            sup_deep = round(self.data['sup_deep'].iloc[i],2) 
            res_r1 = round(self.data['res_r1'].iloc[i],2)
            res_r2 = round(self.data['res_r2'].iloc[i],2)
            res_r3 = round(self.data['res_r3'].iloc[i],2)   
            
            # Calculate scores
            buy_score = 0.0
            sell_score = 0.0
            
            # Trend: Supertrend (Green) +0.4 Directional bias.
            if trend == 'up_trend':
                buy_score += 0.4
            elif trend == 'down_trend':
                sell_score += 0.4
            
            # Momentum: RSI (50 - 70) +0.3 Velocity of the move.
            if momentum_flag == 'rising':
                buy_score += 0.3
            elif momentum_flag == 'falling':
                sell_score += 0.3
            
            # Liquidity: RVOL (> 1.5) +0.3 Institutional participation.
            if volume_flag == 'rising':
                buy_score += 0.3
            elif volume_flag == 'falling':
                sell_score += 0.3
            
            # Volatility: ADX (Rising) +0.2 Confirms the "end of sideways"
            if i > 0 and tradeable:
                if trend == 'up_trend':
                    buy_score += 0.2
                elif trend == 'down_trend':
                    sell_score += 0.2
            
            # Price Action: Breakout / Pattern +0.3 Precise entry/timing.
            if ema_38_channel_flag == 'rising':
                buy_score += 0.3
            elif ema_38_channel_flag == 'falling':
                sell_score += 0.3
            
            #-----------------------------------------------------------------------
            #                    DETERMINE SIGNAL TYPE-STARTS
            #-----------------------------------------------------------------------
            signal_type = ''
            if i >= 5:  # Ensure enough previous records
                # Check for LONG
            
                #if ( and
                if (ema_38_close_flag == 'rising' and
                    ema200_trend_flag == 'rising' and
                    momentum_flag == 'rising' and
                    #trend_change_direction == 'going_up' and
                    (volume_flag == 'rising' or volume_flag == 'normal')
                    #tradeable == True
                    ):
                    signal_type = 'LONG'
                # Check for SHORT
                #elif (tradeable == True and                
                elif (ema_38_close_flag == 'falling' and
                    ema200_trend_flag == 'falling' and
                    momentum_flag == 'falling' and
                    #trend_change_direction == 'going_down' and
                    (volume_flag == 'falling' or volume_flag == 'normal') 
                    #tradeable == True
                    ):
                    signal_type = 'SHORT'
                else:
                    signal_type = ''
            
            signal = {
                'SYMBOL': self.symbol,
                'DATE': self.data.index[i],
                'CLOSE_PRICE': self.data['Close'].iloc[i],
                'FINAL_ACTION': signal_type,
                'COND1_TRADEABLE': tradeable,
                'COND2_EMA_38_CLOSE_FLAG' : ema_38_close_flag,
                'COND3_EMA200_TREND_FLAG': ema200_trend_flag,
                'COND4_MOMENTUM_FLAG': momentum_flag,
                'COND5_VOLUME_FLAG' : volume_flag,
                'EMA_38_CHANNEL_FLAG' : ema_38_channel_flag,
                'EMA200_DIRECTION': ema200_direction,
                'TREND': trend,
                'TREND_CHANGE_DIRECTION': trend_change_direction,                
                'RANGE_OK' : range_ok,
                'FINAL_BUY_SCORE': buy_score,
                'FINAL_SELL_SCORE': sell_score,
                'SUP_S1_LOW': sup_s1_low,
                'SUP_S1_HIGH': sup_s1_high,
                'SUP_DEEP': sup_deep,
                'RES_R1': res_r1,
                'RES_R2': res_r2,
                'RES_R3': res_r3,
                #'exit': exit_trade,
                #'trailing_sl': trailing_sl,
                'INDEX': i,
                #'st_stop_loss': st_stop_loss,
                #'st_resistance': st_resistance,
                #'supertrend_flag': supertrend_flag,                
                'MOMENTUM' : rsi
            }
            #if i in [185, 194, 253, 303, 314, 320, 387, 440, 453, 471]:
            #    print(f"Signal at index {i}: {signal}")
            self.signals.append(signal)

        #print(f"[{self.symbol}] Generated {len(self.signals)} signals")
        return self.signals  

    def run(self, lookback_bars: int = 10) -> List[Dict]:
        """
        Run the complete signal generation process.
        Returns:             List of generated signals
        """
        #--------------------------Step1: Get Price Action Indicators
        self.calculate_ema200()
        self.price_action()   

        # Prepare data for compute_supertrend (expects 'high', 'low', 'close')
        temp_df = self.data[['High', 'Low', 'Close']].rename(columns={'High': 'high','Low': 'low','Close': 'close'})

        #--------------------------Step2: Compute True Range 
        self.data['true_range'] = self.true_candle_range(temp_df)  # Calculate True Range
        tr = self.data['true_range']
        # Average true range for exhaustion check (10 bars is ideal for swing)
        self.data['avg_tr'] = tr.rolling(10).mean()
        # Range guard for current candle
        self.data['range_ok'] = tr <= 1.5 * self.data['avg_tr']
        
        #--------------------------Step3: Compute supertrend and direction
        self.compute_supertrend(temp_df, atr_period=10, factor=3.0)
        #print("### Get Trend Direction")
        #self.data['direction'] = direction
        self.data['is_uptrend'] = self.data['direction'] < 0
        self.data['is_downtrend'] = self.data['direction'] > 0
        self.data['trend'] = np.where(self.data['is_uptrend'], 'up_trend', 
                                      np.where(self.data['is_downtrend'], 'down_trend', '')) 

        #print("### Get Change in Trend")
        self.data['trend_change'] = self.data['direction'] != self.data['direction'].shift(1)
        self.data['down_to_up'] = (self.data['direction'].shift(1) > self.data['direction'])
        self.data['up_to_down'] = (self.data['direction'].shift(1) < self.data['direction'])
        self.data['trend_change_direction'] = np.where(self.data['down_to_up'], 'going_up',
                                                       np.where(self.data['up_to_down'], 'going_down', ''))

        #-------------------------- Apply trade_allowed for tradeable detection
        self.data['tradeable'] = self.trade_allowed(temp_df)         

        #print("### Set Stop Loss for Stocks in up_trend")
        #self.data['supertrend'] = supertrend
        # If trend is uptrend then update the st_stop_loss with supertrend value 
        #self.data['st_stop_loss'] = np.where(self.data['is_uptrend'], self.data['supertrend'], np.nan)   
        # If trend is downtrend then update the st_resistance with supertrend value
        #self.data['st_resistance'] = np.where(self.data['is_downtrend'], self.data['supertrend'], np.nan)
        
        #Step4: Calculate RSI
        self.calculate_rsi(period=14)
        
        #--------------------------Step5: Calculate RVOL
        self.calculate_rvol()

        #--------------------------Step6: Calculate Support & Resistance Levels
        self.calculate_support_n_resistance()        

        #--------------------------Step7: Generate Signals
        return self.generate_signals(lookback_bars)
        
    def get_data_with_indicators(self) -> pd.DataFrame:
        """Return the data with all calculated indicators."""
        return self.data


