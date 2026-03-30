"""
Training script for DDQN trading agent.

This script trains a DDQN agent on stock data and saves results.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from time import time
from typing import Optional, Tuple

from src.trading_env import TradingEnvironment
from src.dqn_agent import DDQNAgent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DDQN trading agent')
    
    # Data parameters
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol')
    parser.add_argument('--trading_days', type=int, default=252,
                        help='Number of days per episode')
    parser.add_argument('--trading_cost', type=float, default=1e-3,
                        help='Trading cost in basis points')
    parser.add_argument('--time_cost', type=float, default=1e-4,
                        help='Time cost per step')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Agent parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Final epsilon')
    parser.add_argument('--epsilon_decay_steps', type=int, default=250,
                        help='Steps for linear epsilon decay')
    parser.add_argument('--epsilon_exp_decay', type=float, default=0.99,
                        help='Exponential decay factor')
    parser.add_argument('--replay_capacity', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for training')
    parser.add_argument('--tau', type=int, default=100,
                        help='Target network update frequency')
    parser.add_argument('--l2_reg', type=float, default=1e-6,
                        help='L2 regularization factor')
    
    # Architecture
    parser.add_argument('--architecture', type=int, nargs='+', default=[256, 256],
                        help='Hidden layer sizes')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
    
    return parser.parse_args()


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def track_results(
    episode: int,
    navs: list,
    market_navs: list,
    diffs: list,
    start_time: float,
    epsilon: float
) -> None:
    """Print training progress."""
    nav_ma_100 = np.mean(navs[-100:]) if navs else 1.0
    nav_ma_10 = np.mean(navs[-10:]) if navs else 1.0
    market_nav_100 = np.mean(market_navs[-100:]) if market_navs else 1.0
    market_nav_10 = np.mean(market_navs[-10:]) if market_navs else 1.0
    win_ratio = np.sum([d > 0 for d in diffs[-100:]]) / min(len(diffs), 100) if diffs else 0.0
    
    elapsed = time() - start_time
    print(f'{episode:4d} | {elapsed:6.1f}s | A:{nav_ma_100-1:6.1%} ({nav_ma_10-1:6.1%}) | '
          f'M:{market_nav_100-1:6.1%} ({market_nav_10-1:6.1%}) | '
          f'W:{win_ratio:5.1%} | ε:{epsilon:6.3f}')


def plot_results(
    results_df: pd.DataFrame,
    output_dir: Path,
    ticker: str
) -> None:
    """Generate and save performance plots."""
    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)
    
    # Rolling returns
    df1 = (results_df[['Agent', 'Market']]
           .sub(1)
           .rolling(100)
           .mean())
    df1.plot(ax=axes[0], title='Annual Returns (Moving Average)', lw=1)
    
    # Win ratio
    df2 = results_df['Strategy Wins (%)'].div(100).rolling(50).mean()
    df2.plot(ax=axes[1], title='Agent Outperformance (%, Moving Average)')
    axes[1].axhline(0.5, ls='--', c='k', lw=1)
    
    # Formatting
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    
    axes[0].set_ylabel('Return')
    axes[1].set_ylabel('Win Rate')
    
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / f'{ticker}_performance.png', dpi=300)
    plt.close()
    
    # Distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(results_df['Difference'], bins=50, ax=ax)
    ax.set_title(f'{ticker}: Distribution of Agent - Market Returns')
    ax.set_xlabel('Excess Return')
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_dir / f'{ticker}_distribution.png', dpi=300)
    plt.close()


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Setup output directory
    output_dir = setup_output_dir(args.output_dir)
    
    print(f"Training DDQN agent on {args.ticker} for {args.episodes} episodes")
    print(f"Episode length: {args.trading_days} days")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Create environment
    env = TradingEnvironment(
        trading_days=args.trading_days,
        trading_cost_bps=args.trading_cost,
        time_cost_bps=args.time_cost,
        ticker=args.ticker
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    print(f"State dimension: {state_dim}, Actions: {n_actions}")
    
    # Create agent
    agent = DDQNAgent(
        state_dim=state_dim,
        num_actions=n_actions,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        epsilon_exponential_decay=args.epsilon_exp_decay,
        replay_capacity=args.replay_capacity,
        architecture=tuple(args.architecture),
        l2_reg=args.l2_reg,
        tau=args.tau,
        batch_size=args.batch_size
    )
    
    # Training loop
    navs = []
    market_navs = []
    diffs = []
    start_time = time()
    
    for episode in range(1, args.episodes + 1):
        state = env.reset(seed=args.seed + episode)
        episode_reward = 0
        step = 0
        
        while True:
            action = agent.epsilon_greedy_policy(state.reshape(1, -1))
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            done = terminated or truncated
            agent.memorize_transition(
                state, action, reward, next_state,
                0.0 if done else 1.0
            )
            
            if agent.train:
                agent.experience_replay()
            
            if done:
                break
            state = next_state
        
        # Store results
        result = env.simulator.result()
        nav = result.nav.iloc[-1]
        market_nav = result.mark
