"""
Main script to train the DDQN agent on the trading environment.
Uses the original TradingEnvironment from trading_env.py.
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import gymnasium as gym
from gymnasium.envs.registration import register

# Add current directory to path to import local modules
sys.path.append(str(Path(__file__).parent))

# Import your original environment and the new agent
from trading_env import TradingEnvironment
from src.agent import DDQNAgent

# Register the trading environment
register(
    id='trading-v0',
    entry_point='trading_env:TradingEnvironment',
    max_episode_steps=252
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a DDQN trading agent.')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--trading-days', type=int, default=252,
                       help='Trading days per episode')
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory to save results')
    return parser.parse_args()


def format_time(t):
    """Format seconds to HH:MM:SS."""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:02.0f}:{m:02.0f}:{s:02.0f}'


def track_results(episode, navs, market_navs, diffs, total_time, epsilon):
    """Print training progress."""
    if episode % 10 != 0:
        return

    nav_ma_100 = np.mean(navs[-100:]) if len(navs) >= 100 else np.mean(navs)
    nav_ma_10 = np.mean(navs[-10:]) if len(navs) >= 10 else np.mean(navs)
    market_nav_100 = np.mean(market_navs[-100:]) if len(market_navs) >= 100 else np.mean(market_navs)
    market_nav_10 = np.mean(market_navs[-10:]) if len(market_navs) >= 10 else np.mean(market_navs)

    win_ratio = np.sum([d > 0 for d in diffs[-100:]]) / min(len(diffs), 100)

    print(f'{episode:>4d}|{format_time(total_time)}|'
          f'A:{nav_ma_100-1:>6.1%}({nav_ma_10-1:>6.1%})|'
          f'M:{market_nav_100-1:>6.1%}({market_nav_10-1:>6.1%})|'
          f'W:{win_ratio:>5.1%}|ε:{epsilon:>6.3f}')


def plot_results(results_df, results_path):
    """Plot and save training results."""
    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

    # Annual returns
    df1 = (results_df[['Agent', 'Market']]
           .sub(1)
           .rolling(100)
           .mean())
    df1.plot(ax=axes[0], title='Annual Returns (Moving Average)', lw=1)

    # Outperformance
    df2 = results_df['Strategy Wins (%)'].div(100).rolling(50).mean()
    df2.plot(ax=axes[1], title='Agent Outperformance (%, Moving Average)')

    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    axes[1].axhline(0.5, ls='--', c='k', lw=1)

    sns.despine()
    fig.tight_layout()
    fig.savefig(results_path / 'performance.png', dpi=300)
    plt.close(fig)


def main():
    """Main training loop."""
    args = parse_args()

    # Set up paths
    results_path = Path(args.results_dir)
    results_path.mkdir(exist_ok=True)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create environment using your original TradingEnvironment
    env = TradingEnvironment(
        trading_days=args.trading_days,
        ticker=args.ticker,
        trading_cost_bps=1e-4,
        time_cost_bps=1e-5
    )

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {num_actions}")
    print(f"Trading days per episode: {args.trading_days}")

    # Hyperparameters
    agent = DDQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=250,
        epsilon_exponential_decay=0.99,
        replay_capacity=int(1e6),
        architecture=(256, 256),
        l2_reg=1e-6,
        tau=100,
        batch_size=4096
    )

    print('\nModel summary:')
    agent.online_network.summary()

    # Training loop
    navs, market_navs, diffs = [], [], []
    print(f'\nStart training for {args.episodes} episodes...')

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.epsilon_greedy_policy(state.reshape(-1, state_dim))
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.memorize_transition(state, action, reward, next_state,
                                     0.0 if terminated else 1.0)
            agent.experience_replay()
            state = next_state

        # Track results
        simulator = env.simulator
        result = simulator.result()
        final = result.iloc[-1]

        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        market_nav = final.market_nav
        market_navs.append(market_nav)

        diff = nav - market_nav
        diffs.append(diff)

        # Print progress
        track_results(episode, navs, market_navs, diffs,
                     agent.total_steps, agent.epsilon)

    env.close()

    # Save results
    results_df = pd.DataFrame({
        'Episode': list(range(1, episode + 1)),
        'Agent': navs,
        'Market': market_navs,
        'Difference': diffs
    }).set_index('Episode')
    results_df['Strategy Wins (%)'] = (results_df.Difference > 0).rolling(100).sum()
    results_df.to_csv(results_path / 'results.csv')

    # Plot results
    plot_results(results_df, results_path)

    print(f'\nTraining complete. Results saved to {results_path}')
    print(f'  - results.csv: training history')
    print(f'  - performance.png: performance plot')


if __name__ == '__main__':
    main()
