"""
Trading Environment for Reinforcement Learning.

This module implements a trading environment with:
- Three actions: SHORT, HOLD, LONG
- Transaction costs and time costs
- Buy-and-hold benchmark
"""

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from sklearn.preprocessing import scale
import yfinance as yf
from typing import Tuple, Optional, List, Dict, Any

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class DataSource:
    """
    Data source for TradingEnvironment.
    
    Loads and preprocesses daily price and volume data from Yahoo Finance.
    Provides data for each new episode with random start date.
    
    Attributes:
        ticker (str): Stock symbol (e.g., 'AAPL').
        trading_days (int): Number of days per episode.
        normalize (bool): Whether to normalize features.
        data (pd.DataFrame): Preprocessed data with features.
        min_values (pd.Series): Minimum values for normalization.
        max_values (pd.Series): Maximum values for normalization.
    """
    
    def __init__(self, trading_days: int, ticker: str = 'AAPL', normalize: bool = True) -> None:
        """
        Initialize the data source.
        
        Args:
            trading_days: Number of trading days per episode.
            ticker: Stock symbol to download.
            normalize: If True, scale features to zero mean and unit variance.
        """
        self.ticker = ticker
        self.trading_days = trading_days
        self.normalize = normalize
        self.data = self.load_data()
        self.preprocess_data()
        self.min_values = self.data.min()
        self.max_values = self.data.max()
        self.step = 0
        self.offset = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance.
        
        Returns:
            DataFrame with MultiIndex (date, ticker) and columns:
            close, volume, low, high.
        """
        try:
            df = yf.download(self.ticker, start='2000-01-01', end='2023-12-31', progress=False)
            df = df[['Adj Close', 'Volume', 'Low', 'High']]
            df.columns = ['close', 'volume', 'low', 'high']
            df.index = pd.MultiIndex.from_product([df.index, [self.ticker]], names=['date', 'ticker'])
            log.info(f"Loaded {len(df)} days of data for {self.ticker}")
            return df
        except Exception as e:
            log.error(f"Failed to load data: {e}")
            raise
    
    def preprocess_data(self) -> None:
        """
        Calculate technical indicators and normalize data.
        
        Computes:
            - returns: daily returns
            - ret_2, ret_5, ret_10, ret_21: multi-period returns
            - rsi: Relative Strength Index (14-day)
            - macd: Moving Average Convergence Divergence
            - atr: Average True Range (14-day)
        """
        # Returns
        self.data['returns'] = self.data.close.pct_change()
        self.data['ret_2'] = self.data.close.pct_change(2)
        self.data['ret_5'] = self.data.close.pct_change(5)
        self.data['ret_10'] = self.data.close.pct_change(10)
        self.data['ret_21'] = self.data.close.pct_change(21)
        
        # RSI (14)
        delta = self.data['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (12,26,9)
        ema12 = self.data['close'].ewm(span=12, adjust=False).mean()
        ema26 = self.data['close'].ewm(span=26, adjust=False).mean()
        self.data['macd'] = ema12 - ema26
        
        # ATR (14)
        high_low = self.data['high'] - self.data['low']
        high_close = abs(self.data['high'] - self.data['close'].shift())
        low_close = abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['atr'] = tr.rolling(window=14).mean()
        
        # Clean up
        self.data = (self.data.replace((np.inf, -np.inf), np.nan)
                     .drop(['high', 'low', 'close', 'volume'], axis=1)
                     .dropna())
        
        # Store returns before normalization
        r = self.data.returns.copy()
        
        # Normalize features (except returns)
        if self.normalize:
            features = self.data.drop(columns=['returns'])
            self.data[features.columns] = scale(features)
        
        # Reorder columns: returns first
        features = self.data.columns.drop('returns')
        self.data['returns'] = r
        self.data = self.data.loc[:, ['returns'] + list(features)]
        log.info(f"Preprocessed data shape: {self.data.shape}")
    
    def reset(self) -> None:
        """
        Reset the data source for a new episode.
        
        Randomly selects a starting index within the available data range.
        """
        high = len(self.data.index) - self.trading_days
        if high <= 0:
            raise ValueError(f"Not enough data: need {self.trading_days} days, have {len(self.data.index)}")
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0
    
    def take_step(self) -> Tuple[np.ndarray, bool]:
        """
        Get observation for the current day.
        
        Returns:
            obs: Feature vector for the current day.
            done: True if episode has ended.
        """
        obs = self.data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.trading_days
        return obs.astype(np.float32), done


class TradingSimulator:
    """
    Trading simulator for single-instrument trading.
    
    Tracks positions, trades, costs, and Net Asset Value (NAV).
    Compares agent performance against buy-and-hold benchmark.
    
    Actions mapping:
        0: SHORT (position = -1)
        1: HOLD (position = 0)
        2: LONG (position = 1)
    """
    
    def __init__(self, steps: int, trading_cost_bps: float, time_cost_bps: float) -> None:
        """
        Initialize the trading simulator.
        
        Args:
            steps: Number of steps per episode.
            trading_cost_bps: Trading cost in basis points (1bps = 0.01%).
            time_cost_bps: Time cost per step when holding a position.
        """
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        self.reset()
    
    def reset(self) -> None:
        """Reset all simulator state for a new episode."""
        self.step = 0
        self.actions = np.zeros(self.steps, dtype=np.int32)
        self.navs = np.ones(self.steps, dtype=np.float32)
        self.market_navs = np.ones(self.steps, dtype=np.float32)
        self.strategy_returns = np.zeros(self.steps, dtype=np.float32)
        self.positions = np.zeros(self.steps, dtype=np.float32)
        self.costs = np.zeros(self.steps, dtype=np.float32)
        self.trades = np.zeros(self.steps, dtype=np.float32)
        self.market_returns = np.zeros(self.steps, dtype=np.float32)
    
    def take_step(self, action: int, market_return: float) -> Tuple[float, Dict[str, Any]]:
        """
        Process a trading step.
        
        Args:
            action: Agent's action (0=SHORT, 1=HOLD, 2=LONG).
            market_return: Current day's market return.
            
        Returns:
            reward: Reward for this step.
            info: Dictionary with step information.
        """
        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action
        
        end_position = action - 1  # Map 0→-1, 1→0, 2→1
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades
        
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        
        reward = start_position * market_return - self.costs[self.step]
        self.strategy_returns[self.step] = reward
        
        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])
        
        info = {
            'reward': reward,
            'nav': float(self.navs[self.step]),
            'costs': float(self.costs[self.step]),
            'position': float(end_position)
        }
        
        self.step += 1
        return reward, info
    
    def result(self) -> pd.DataFrame:
        """
        Get current state as DataFrame.
        
        Returns:
            DataFrame with columns: action, nav, market_nav, market_return,
            strategy_return, position, cost, trade.
        """
        return pd.DataFrame({
            'action': self.actions,
            'nav': self.navs,
            'market_nav': self.market_navs,
            'market_return': self.market_returns,
            'strategy_return': self.strategy_returns,
            'position': self.positions,
            'cost': self.costs,
            'trade': self.trades
        })


class TradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning.
    
    A simple trading environment with three actions (SHORT, HOLD, LONG).
    Episode length is fixed to trading_days. Each step provides:
        - Observation: feature vector of market data
        - Reward: position return minus transaction costs
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        trading_days: int,
        trading_cost_bps: float = 1e-3,
        time_cost_bps: float = 1e-4,
        ticker: str = 'AAPL'
    ) -> None:
        """
        Initialize the trading environment.
        
        Args:
            trading_days: Number of steps per episode.
            trading_cost_bps: Trading cost in basis points.
            time_cost_bps: Time cost per step when holding a position.
            ticker: Stock symbol to trade.
        """
        super().__init__()
        
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.ticker = ticker
        self.time_cost_bps = time_cost_bps
        
        self.data_source = DataSource(
            trading_days=self.trading_days,
            ticker=ticker
        )
        self.simulator = TradingSimulator(
            steps=self.trading_days,
            trading_cost_bps=self.trading_cost_bps,
            time_cost_bps=self.time_cost_bps
        )
        
        self.action_space = spaces.Discrete(3)
        
        # Observation space bounds
        obs_low = self.data_source.min_values.to_numpy()
        obs_high = self.data_source.max_values.to_numpy()
        self.observation_space = spaces.Box(
            low=obs_low.astype(np.float32),
            high=obs_high.astype(np.float32),
            dtype=np.float32
        )
        
        self._reset_called = False
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0=SHORT, 1=HOLD, 2=LONG).
            
        Returns:
            observation: Next state observation.
            reward: Reward for this step.
            terminated: True if episode ended.
            truncated: True if episode was truncated.
            info: Additional information.
        """
        if not self._reset_called:
            raise RuntimeError("Call reset() before step()")
            
        assert self.action_space.contains(action), f'Invalid action: {action}'
        
        observation, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(
            action=action,
            market_return=observation[0]
        )
        
        return observation, reward, done, False, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Initial observation.
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data_source.reset()
        self.simulator.reset()
        self._reset_called = True
        
        obs, _ = self.data_source.take_step()
        return obs
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment (not implemented)."""
        pass
    
    def close(self) -> None:
        """Close the environment."""
        self._reset_called = False
