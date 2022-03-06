from utilities.environment.trading_env import TradingEnv
from utilities.rl.deep_q_network import DeepQNetwork
from utilities.io.stock_data_generator import StockDataGenerator
from utilities.nn.neural_network import NeuralNetwork
from utilities.rl.deep_q_network import DeepQNetwork
from utilities.environment.quandl_env_src import QuandlEnvSrc
from utilities.utils.checks import check_computer_device
from utilities.rl.ddqn_agent import DDQNAgent
from time import time
import logging
from dotenv import load_dotenv
load_dotenv()

from utilities.utils.checks import track_results
# from utilities.rl.policy_gradient import PolicyGradient
import tensorflow as tf
import gym
import pandas as pd
import numpy as np
from gym.envs.registration import register
import os

NUM_ACTIONS = 3 # [buy, sell, hold]
NUM_OBSERVATIONS = 4 # Deve ser 

trading_days = 252
# register(
#     id='trading-v0',
#     entry_point='gym_trading.envs.trading_env:TradingEnv',
#     timestep_limit=trading_days,
# )
def routine():

    logging.info("Running the routine")
    check_computer_device()
    
    
    quandl_env_src = QuandlEnvSrc(days=trading_days)
    trading_environment = TradingEnv(quandl_env_src)
    trading_environment.timestep_limit=trading_days
    trading_environment.seed(42)

    ### Get Environment Params
    state_dim = trading_environment.observation_space.shape[0]
    num_actions = trading_environment.action_space.n
    # max_episode_steps = trading_environment.spec.max_episode_steps
    max_episode_steps = 252

    ## Define hyperparameters
    gamma = .99,  # discount factor
    tau = 100  # target network update frequency

    ### NN Architecture
    architecture = (64, 256, 64)  # units per layer
    learning_rate = 0.0001  # learning rate
    l2_reg = 1e-6  # L2 regularization

    ### Experience Replay
    replay_capacity = int(1e6)
    batch_size = 4096

    ### $\epsilon$-greedy Policy
    epsilon_start = 1.0
    epsilon_end = .01
    epsilon_decay_steps = 250
    epsilon_exponential_decay = .99

    ## Create DDQN Agent
    # We will use [TensorFlow](https://www.tensorflow.org/) to create our Double Deep Q-Network .
    tf.keras.backend.clear_session()
    
    ddqn = DDQNAgent(state_dim=state_dim,
                 num_actions=num_actions,
                 learning_rate=learning_rate,
                 gamma=gamma,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                 epsilon_decay_steps=epsilon_decay_steps,
                 epsilon_exponential_decay=epsilon_exponential_decay,
                 replay_capacity=replay_capacity,
                 architecture=architecture,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size)
    
    ddqn.online_network.summary()

    ## Run Experiment
    ### Set parameters

    total_steps = 0
    max_episodes = 1000    

    ### Initialize variables

    episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

    # ddqn.training()

    start = time()
    results = []
    for episode in range(1, max_episodes + 1):
        this_state = trading_environment.reset()
        for episode_step in range(max_episode_steps):
            action = ddqn.epsilon_greedy_policy(this_state.to_numpy().reshape(-1, state_dim))
            next_state, reward, done, _ = trading_environment.step(action)
        
            ddqn.memorize_transition(this_state, 
                                    action, 
                                    reward, 
                                    next_state, 
                                    0.0 if done else 1.0)
            if ddqn.train:
                ddqn.experience_replay()
            if done:
                break
            this_state = next_state

        # get DataFrame with sequence of actions, returns and nav values
        result = ddqn.result() # trading_environment.env.simulator.result()
        # get results of last step
        final = result.iloc[-1]

        # apply return (net of cost) of last action to last starting nav 
        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        # market nav 
        market_nav = final.market_nav
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
                        time() - start, ddqn.epsilon)
        if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
            print(result.tail())
            break

    trading_environment.close()

    results = pd.DataFrame({'Episode': list(range(1, episode+1)),
                        'Agent': navs,
                        'Market': market_navs,
                        'Difference': diffs}).set_index('Episode')

    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
    results.info()



    ################################################
    ##########--------SEGUNDO MODO--------##########
    ################################################
    # input_shape = (None, NUM_OBSERVATIONS)
    # output_action = NUM_ACTIONS

    # trading_cost_bps = 1e-3
    # time_cost_bps = 1e-4

    # nn = NeuralNetwork()
    
    # model = nn.build(input_shape, output_action)
    
    # dqn = DeepQNetwork()
    
    # env = TradingEnv()
    # env.timestep_limit=1000
    
    # observation = env.reset()
    # done = False
    # navs = []
    # while not done:
    #     action = 1 # stay flat
    #     observation, reward, done, info = env.step(action)
    #     navs.append(info['nav'])
    #     if done:
    #         print(f'Annualized return: {navs[len(navs)-1]-1}')
    #         pd.DataFrame(navs).plot()

    # #import trading_env as te

    # stayflat     = lambda o,e: 1   # stand pat
    # buyandhold   = lambda o,e: 2   # buy on day #1 and hold
    # randomtrader = lambda o,e: e.action_space.sample() # retail trader

    # # to run singly, we call run_strat.  we are returned a dataframe containing 
    # #  all steps in the sim.
    # bhdf = env.run_strat(buyandhold)

    # print(bhdf.head())

    # # we can easily plot our nav in time:
    # bhdf.bod_nav.plot(title='buy & hold nav')

    # env.run_strat(buyandhold).bod_nav.plot(title='same strat, different results')
    # env.run_strat(buyandhold).bod_nav.plot()
    # env.run_strat(buyandhold).bod_nav.plot()


    # # create the tf session
    # tf.compat.v1.reset_default_graph() 
    # sess1 = tf.compat.v1.InteractiveSession()
    # # create policygradient
    # pg = PolicyGradient(sess1, obs_dim=5, num_actions=3, learning_rate=1e-2 )

    # # and now let's train it and evaluate its progress.  NB: this could take some time...
    # df,sf = pg.train_model( env,episodes=25001, log_freq=100)#, load_model=True)