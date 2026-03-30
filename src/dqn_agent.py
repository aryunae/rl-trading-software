"""
Double Deep Q-Network (DDQN) Agent for Reinforcement Learning trading.

This module implements a DDQN agent with experience replay, target network,
and epsilon-greedy policy for exploration/exploitation trade-off.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2
from collections import deque
from random import sample
from typing import List, Tuple, Optional, Any, Union


class DDQNAgent:
    """
    Double Deep Q-Network Agent for reinforcement learning tasks.
    
    This agent uses two networks (online and target) to reduce overestimation bias
    in Q-learning. It stores experiences in a replay buffer and learns from
    random batches to break correlations between consecutive samples.
    
    Attributes:
        state_dim (int): Dimension of the state space.
        num_actions (int): Number of possible actions.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Current exploration rate.
        replay_capacity (int): Maximum size of experience replay buffer.
        batch_size (int): Number of samples per training batch.
        tau (int): Target network update frequency.
        online_network (tf.keras.Model): Network used for action selection.
        target_network (tf.keras.Model): Network used for target Q-value computation.
        experience (deque): Replay buffer storing (s, a, r, s', done) tuples.
        losses (List): History of training losses.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        epsilon_exponential_decay: float,
        replay_capacity: int,
        architecture: Tuple[int, ...],
        l2_reg: float,
        tau: int,
        batch_size: int
    ) -> None:
        """
        Initialize the DDQN agent.
        
        Args:
            state_dim: Dimension of the state space.
            num_actions: Number of possible actions.
            learning_rate: Learning rate for Adam optimizer.
            gamma: Discount factor (0 to 1) for future rewards.
            epsilon_start: Initial exploration rate.
            epsilon_end: Final exploration rate after decay.
            epsilon_decay_steps: Number of steps for linear epsilon decay.
            epsilon_exponential_decay: Multiplicative factor after linear decay.
            replay_capacity: Maximum size of experience replay buffer.
            architecture: Tuple of hidden layer sizes.
            l2_reg: L2 regularization factor for network weights.
            tau: Number of steps between target network updates.
            batch_size: Number of samples per training batch.
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg
        
        # Networks
        self.online_network = self._build_model(trainable=True)
        self.target_network = self._build_model(trainable=False)
        self._update_target()
        
        # Epsilon decay parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(epsilon_decay_steps, 1)
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history: List[float] = []
        
        # Training statistics
        self.total_steps = 0
        self.train_steps = 0
        self.episodes = 0
        self.episode_length = 0
        self.episode_reward = 0.0
        self.rewards_history: List[float] = []
        self.steps_per_episode: List[int] = []
        
        # Experience replay
        self.experience = deque(maxlen=replay_capacity)
        self.batch_size = batch_size
        self.tau = tau
        self.losses: List[float] = []
        self.idx = tf.range(batch_size)
        self.train = True
    
    def _build_model(self, trainable: bool = True) -> Sequential:
        """
        Build the neural network model for Q-value approximation.
        
        Args:
            trainable: Whether the model weights are trainable.
            
        Returns:
            Compiled Keras Sequential model.
        """
        layers = []
        n = len(self.architecture)
        
        for i, units in enumerate(self.architecture, 1):
            layers.append(
                Dense(
                    units=units,
                    input_dim=self.state_dim if i == 1 else None,
                    activation='relu',
                    kernel_regularizer=l2(self.l2_reg),
                    name=f'Dense_{i}',
                    trainable=trainable
                )
            )
        
        layers.append(Dropout(0.1))
        layers.append(
            Dense(
                units=self.num_actions,
                trainable=trainable,
                name='Output'
            )
        )
        
        model = Sequential(layers)
        model.compile(
            loss='mean_squared_error',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def _update_target(self) -> None:
        """Copy weights from online network to target network."""
        self.target_network.set_weights(self.online_network.get_weights())
    
    def epsilon_greedy_policy(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon, selects a random action for exploration.
        Otherwise selects the action with highest Q-value.
        
        Args:
            state: Current state observation.
            
        Returns:
            Selected action index (0 to num_actions-1).
        """
        self.total_steps += 1
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        
        q_values = self.online_network.predict(state, verbose=0)
        return int(np.argmax(q_values, axis=1).squeeze())
    
    def memorize_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        not_done: float
    ) -> None:
        """
        Store a transition in the replay buffer and update episode statistics.
        
        Args:
            state: Current state before action.
            action: Action taken.
            reward: Reward received.
            next_state: State after action.
            not_done: 1.0 if episode not terminated, 0.0 otherwise.
        """
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            # Episode ended - update epsilon and stats
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon = max(
                        self.epsilon_end,
                        self.epsilon - self.epsilon_decay
                    )
                else:
                    self.epsilon *= self.epsilon_exponential_decay
                    self.epsilon = max(self.epsilon_end, self.epsilon)
            
            self.epsilon_history.append(self.epsilon)
            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward = 0.0
            self.episode_length = 0
        
        self.experience.append((state, action, reward, next_state, not_done))
    
    def experience_replay(self) -> None:
        """
        Sample a random batch from replay buffer and update the network.
        
        Implements Double DQN update rule:
        - Online network selects best action for next state
        - Target network evaluates that action's value
        - Loss = MSE(target - predicted_q)
        """
        if self.batch_size > len(self.experience):
            return
        
        # Sample random batch
        minibatch = map(
            np.array,
            zip(*sample(self.experience, self.batch_size))
        )
        states, actions, rewards, next_states, not_done = minibatch
        
        # Reshape for batch processing
        if len(states.shape) == 1:
            states = states.reshape(-1, self.state_dim)
            next_states = next_states.reshape(-1, self.state_dim)
        
        # Online network Q-values for next states
        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)
        
        # Target network Q-values for next states
        next_q_values_target = self.target_network.predict_on_batch(next_states)
        
        # Create indices for gathering
        batch_indices = tf.range(self.batch_size)
        target_q_values = tf.gather_nd(
            next_q_values_target,
            tf.stack(
                (batch_indices, tf.cast(best_actions, tf.int32)),
                axis=1
            )
        )
        
        # Compute targets
        targets = rewards + not_done * self.gamma * target_q_values.numpy()
        
        # Current Q-values
        q_values = self.online_network.predict_on_batch(states)
        
        # Update Q-values for taken actions
        for i in range(self.batch_size):
            q_values[i, actions[i]] = targets[i]
        
        # Gradient update
        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)
        
        # Update target network periodically
        if self.total_steps % self.tau == 0:
            self._update_target()
    
    def save_model(self, path: str) -> None:
        """Save the online network model to disk."""
        self.online_network.save(path)
    
    def load_model(self, path: str) -> None:
        """Load a saved model into the online network."""
        self.online_network = tf.keras.models.load_model(path)
        self._update_target()
    
    def get_epsilon(self) -> float:
        """Return current epsilon value."""
        return self.epsilon
    
    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            'episodes': self.episodes,
            'total_steps': self.total_steps,
            'avg_reward': np.mean(self.rewards_history[-100:]) if self.rewards_history else 0,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'epsilon': self.epsilon
        }
