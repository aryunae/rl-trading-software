# RL Trading Bot

Reinforcement Learning agent for stock trading using Double Deep Q-Network (DDQN).

## Project Overview

This project implements a reinforcement learning agent that learns to trade a single stock using a custom OpenAI Gym environment. The agent can take three actions: SHORT, HOLD, and LONG. It receives observations including price returns and technical indicators (RSI, MACD, ATR, etc.).

The environment simulates trading with realistic costs (trading and time costs) and tracks both the agent's net asset value (NAV) and a buy-and-hold market benchmark.

## Features

- Custom OpenAI Gym trading environment
- Double DQN agent with experience replay
- Technical indicators calculated with TA-Lib
- Configurable hyperparameters
- Training progress tracking and visualization
- CI/CD pipeline with GitHub Actions

## Installation

### Prerequisites

- Python 3.10 or higher
- TA-Lib system library

### Step 1: Clone the repository

```bash
git clone https://github.com/aryunae/rl-trading-software.git
cd rl-trading-software

### Step 2: Install TA-Lib (system dependency)
Ubuntu/Debian:

bash
sudo apt-get update
sudo apt-get install -y ta-lib
macOS:

bash
brew install ta-lib
Windows:

Download the appropriate .whl file from here and install with pip:

bash
pip install TA_Lib‑0.4.32‑cp310‑cp310‑win_amd64.whl
### Step 3: Install Python dependencies
bash
pip install -r requirements.txt
### Step 4: Prepare data
Place your assets.h5 file in the data/ folder. Create the folder if it doesn't exist:

bash
mkdir -p data
# Copy your assets.h5 file into the data/ folder
Usage
Training the agent
To train the agent with default settings:

bash
python train.py
Training with custom parameters
bash
python train.py --episodes 500 --trading-days 126 --ticker GOOGL
Command-line arguments
Argument	Description	Default
--episodes	Number of episodes to train	1000
--trading-days	Trading days per episode	252
--ticker	Stock ticker symbol	AAPL
--results-dir	Directory to save results	results
Running the notebook
For interactive analysis and visualization, use the original Jupyter notebook:

bash
jupyter notebook 4_q_learning_for_trading.ipynb
Project Structure
text
rl-trading-software/
├── .github/workflows/   # CI/CD configuration
│   └── ci.yml
├── src/                 # Source code
│   ├── __init__.py
│   └── agent.py         # DDQN agent class
├── data/                # Data files (place assets.h5 here)
├── results/             # Training results (auto-generated)
├── trading_env.py       # Trading environment (original)
├── 4_q_learning_for_trading.ipynb  # Jupyter notebook (original)
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore file
└── README.md           # This file
Results
After training, the script will:

Save the training history to results/results.csv

Generate a performance plot at results/performance.png

Display training progress in the console

Example Output
text
Start training for 1000 episodes...
  10|00:00:03|A:-39.5%(-39.5%)|M:  5.6%(  5.6%)|W:20.0%|ε: 0.960
  20|00:01:27|A:-34.8%(-30.0%)|M: 23.7%( 41.8%)|W:15.0%|ε: 0.921
  ...
 990|10:21:10|A: 62.4%( 35.9%)|M: 19.9%( -7.9%)|W:61.0%|ε: 0.000
1000|10:28:18|A: 62.1%( 31.0%)|M: 17.6%(  3.5%)|W:60.0%|ε: 0.000
Training complete. Results saved to results
Performance Plot
The generated performance.png shows:

Left plot: Moving average of annual returns for agent and market

Right plot: Rolling 50-episode average of agent outperformance

Code Quality
This project follows best practices:

Type hints for all function arguments and return values

Docstrings for all classes and methods

Linting with flake8 in CI/CD pipeline

Reproducible results with fixed random seeds

CI/CD Pipeline
GitHub Actions automatically runs on every push and pull request:

Installs system dependencies (TA-Lib)

Installs Python dependencies

Runs a quick training test (5 episodes, 20 trading days)

Runs flake8 linter to check code quality
