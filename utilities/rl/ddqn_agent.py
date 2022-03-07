from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from random import sample
from time import time

from utilities.nn.neural_network import NeuralNetwork
from utilities.utils.checks import track_results
import numpy as np
import pandas as pd

class DDQNAgent:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_model(self, trainable=True):
        
        n = len(self.architecture)

        neural_network = NeuralNetwork(
            self.state_dim, 
            self.num_actions, 
            self.architecture, 
            self.learning_rate, 
            self.l2_reg)

        model = neural_network.build()

        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict_on_batch(states)
        q_values[[self.idx, actions]] = targets

        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()

    
        
    def training(self, env, ):
        
        trading_environment = env
        total_steps = 0
        max_episodes = 1000    
        max_episode_steps =252
        ### Initialize variables

        episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

        # ddqn.training()

        start = time()
        results = []
        
        for episode in range(1, max_episodes + 1):
            step_actions = []
            step_navs = []
            step_mkt_navs = []
            step_strategy_return = []
            this_state = trading_environment.reset()
            for episode_step in range(max_episode_steps):
                action = self.epsilon_greedy_policy(this_state.to_numpy().reshape(-1, self.state_dim))
                next_state, reward, done, info = trading_environment.step(action)
            
                self.memorize_transition(this_state, 
                                        action, 
                                        reward, 
                                        next_state, 
                                        0.0 if done else 1.0)
                step_actions.append(action)
                step_navs.append(info['nav'])
                step_mkt_navs.append(info['mkt_nav'])
                step_strategy_return.append(info['strategy_return'])

                if self.train:
                    self.experience_replay()
                if done:
                    break
                this_state = next_state

            nav =  step_navs[-1] * (1 + step_strategy_return[-1])

            navs.append(nav)

            market_nav = step_mkt_navs[-1]
            market_navs.append(market_nav)

            # track difference between agent an market NAV results
            diff = nav - market_nav
            diffs.append(diff)
            
            if episode % 10 == 0:
                track_results(episode,  
                            # show mov. average results for 100 (10) periods
                            np.mean(navs[-100:]), 
                            np.mean(navs[-10:]), 
                            np.mean(market_navs[-100:]), 
                            np.mean(market_navs[-10:]), 
                            # share of agent wins, defined as higher ending nav
                            np.sum([s > 0 for s in diffs[-100:]])/min(len(diffs), 100), 
                            time() - start, self.epsilon)
            if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
                # print(result.tail())
                break

        trading_environment.close()

        results = pd.DataFrame({'Episode': list(range(1, episode+1)),
                            'Agent': navs,
                            'Market': market_navs,
                            'Difference': diffs}).set_index('Episode')

        results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
        results.info()