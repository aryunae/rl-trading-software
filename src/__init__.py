"""
RL Trading Software Complex

Reinforcement Learning agent for algorithmic trading using Double Deep Q-Network (DDQN).
"""

__version__ = "1.0.0"
__author__ = "aryunae"
__email__ = "endonova.aryuna@gmail.com"

from src.dqn_agent import DDQNAgent
from src.trading_env import TradingEnvironment, TradingSimulator, DataSource
