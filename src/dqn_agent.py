"""
DDQN Agent for reinforcement learning.

Implements a Double Deep Q-Network with experience replay and target network.
"""

import numpy as np
from collections import deque
from random import sample
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from typing import List, Tuple, Deque


class DDQNAgent:
    """Double DQN agent with experience replay."""

    def __init__(self,
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
                 batch_size: int):
        """
        Initialize the DDQN Agent.

        Args:
            state_dim: Dimension of the state space.
            num_actions: Number of possible actions.
            learning_rate: Learning rate for the optimizer.
            gamma: Discount factor.
            epsilon_start: Starting value for epsilon (exploration rate).
            epsilon_end: Final value for epsilon.
            epsilon_decay_steps: Number of steps for linear decay.
            epsilon_exponential_decay: Factor for exponential decay after linear decay.
            replay_capacity: Maximum size of the replay buffer.
            architecture: Tuple of neurons per hidden layer.
            l2_reg: L2 regularization factor.
            tau: Target network update frequency.
            batch_size: Batch size for experience replay.
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience: Deque = deque(maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        # Networks
        self.online_network = self._build_model(trainable=True)
        self.target_network = self._build_model(trainable=False)
        self._update_target()

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history: List[float] = []

        # Training statistics
        self.total_steps = 0
        self.train_steps = 0
        self.episodes = 0
        self.episode_length = 0
        self.episode_reward = 0
        self.rewards_history: List[float] = []
        self.steps_per_episode: List[int] = []

        # Replay parameters
        self.batch_size = batch_size
        self.tau = tau
        self.losses: List[float] = []
        self.idx = tf.range(batch_size)

    def _build_model(self, trainable: bool = True) -> Sequential:
        """Build and compile the neural network model."""
        model = Sequential()
        for i, units in enumerate(self.architecture, 1):
            model.add(Dense(
                units=units,
                activation='relu',
                kernel_regularizer=l2(self.l2_reg),
                input_dim=self.state_dim if i == 1 else None,
                name=f'Dense_{i}',
                trainable=trainable
            ))
        model.add(Dropout(0.1))
        model.add(Dense(
            units=self.num_actions,
            trainable=trainable,
            name='Output'
        ))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _update_target(self):
        """Update target network weights with online network weights."""
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state: np.ndarray) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state.

        Returns:
            int: Chosen action.
        """
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q_values = self.online_network.predict(state, verbose=0)
        return int(np.argmax(q_values, axis=1).squeeze())

    def memorize_transition(self, s: np.ndarray, a: int, r: float,
                           s_prime: np.ndarray, not_done: float):
        """
        Store a transition in the replay buffer and update episode stats.

        Args:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s_prime: Next state.
            not_done: 1.0 if episode not done, else 0.0.
        """
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            # Update epsilon when episode ends
            if self.episodes < self.epsilon_decay_steps:
                self.epsilon = max(0.01, self.epsilon - self.epsilon_decay)
            else:
                self.epsilon = max(0.01, self.epsilon * self.epsilon_exponential_decay)

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        """
        Perform experience replay to update the online network.
        Uses Double DQN formula to compute targets.
        """
        if len(self.experience) < self.batch_size:
            return

        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        # Get Q-values from online network for the next states
        next_q_values_online = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values_online, axis=1)

        # Get Q-values from target network for the next states
        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(
            next_q_values_target,
            tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1)
        )

        # Compute targets
        targets = rewards + not_done * self.gamma * target_q_values

        # Get current Q-values and update only the taken actions
        current_q_values = self.online_network.predict_on_batch(states)
        current_q_values[self.idx.numpy(), actions] = targets

        # Train the network
        loss = self.online_network.train_on_batch(x=states, y=current_q_values)
        self.losses.append(loss)

        # Update target network periodically
        if self.total_steps % self.tau == 0:
            self._update_target()
