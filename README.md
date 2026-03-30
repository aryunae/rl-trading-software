# RL Trading Software Complex

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/YOUR_USERNAME/rl-trading-software/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/rl-trading-software/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Reinforcement Learning agent for algorithmic trading using **Double Deep Q-Network (DDQN)**.

##  Overview

This project implements a complete software complex for training and evaluating a trading agent using reinforcement learning. The agent learns to make trading decisions (SHORT, HOLD, LONG) based on historical stock data and technical indicators, aiming to maximize returns while accounting for transaction costs.

### Key Features

-  **DDQN Agent**: Double Deep Q-Network with experience replay and target network
-  **Trading Environment**: Custom OpenAI Gym environment with realistic transaction costs
-  **Technical Indicators**: RSI, MACD, ATR, and multi-period returns
-  **Real Data**: Automatic data download via yfinance
-  **Comprehensive Tests**: Unit tests with pytest
-  **CI/CD**: Automated testing with GitHub Actions
-  **Fully Documented**: Complete docstrings and type hints

### Actions

| Action | Code | Position | Description |
|--------|------|----------|-------------|
| SHORT | 0 | -1 | Take a short position (bet on price decrease) |
| HOLD | 1 | 0 | Hold cash, no position |
| LONG | 2 | +1 | Take a long position (bet on price increase) |

### Costs

- **Trading cost**: 10 basis points (0.1%) per trade
- **Time cost**: 1 basis point (0.01%) per step when holding a position

##  Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rl-trading-software.git
cd rl-trading-software

### Step 2: Create virtual environment (recommended)

# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

### Step 3: Install dependencies

pip install -r requirements.txt

### Usage
Basic Training
Train the agent on Apple stock (AAPL) for 500 episodes:

python src/train.py --ticker AAPL --episodes 500

Advanced Training with Custom Parameters

python src/train.py \
    --ticker MSFT \
    --episodes 1000 \
    --trading_days 126 \
    --learning_rate 0.0001 \
    --gamma 0.99 \
    --epsilon_start 1.0 \
    --epsilon_end 0.01 \
    --architecture 256 256 \
    --batch_size 4096 \
    --save_model


Command Line Arguments
Argument	Default	Description
--ticker	AAPL	Stock ticker symbol
--trading_days	252	Number of days per episode
--episodes	500	Number of training episodes
--learning_rate	1e-4	Learning rate for optimizer
--gamma	0.99	Discount factor for future rewards
--epsilon_start	1.0	Initial exploration rate
--epsilon_end	0.01	Final exploration rate
--architecture	256 256	Hidden layer sizes
--batch_size	4096	Batch size for training
--save_model	False	Save trained model to disk
--output_dir	./results	Directory for saving results


### Output Files
After training, the following files are saved in ./results/:

results.csv: Episode-by-episode performance metrics

{ticker}_performance.png: Learning curves (rolling returns and win rate)

{ticker}_distribution.png: Distribution of excess returns

model.h5: Trained model (if --save_model is used)

Testing
Run the test suite with pytest:

# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term --cov-report=html


Test Structure

tests/
├── test_environment.py    # Tests for trading environment
└── test_agent.py          # Tests for DDQN agent
