from utilities.env.trading_env import TradingEnv
from utilities.rl.deep_q_network import DeepQNetwork
from utilities.io.stock_data_generator import StockDataGenerator
from utilities.nn.neural_network import NeuralNetwork
from utilities.rl.deep_q_network import DeepQNetwork
from utilities.utils.checks import check_computer_device

import logging
from dotenv import load_dotenv
load_dotenv()

from utilities.rl.policy_gradient import PolicyGradient
import tensorflow as tf

import pandas as pd

NUM_ACTIONS = 3 # [buy, sell, hold]
NUM_OBSERVATIONS = 4 # Deve ser 

def routine():
    logging.info("Running the routine")
    check_computer_device()
    
    input_shape = (None, NUM_OBSERVATIONS)
    output_action = NUM_ACTIONS
    # nn = NeuralNetwork()
    
    # model = nn.build(input_shape, output_action)
    
    # dqn = DeepQNetwork()
    
    env = TradingEnv()
    env.timestep_limit=1000
    
    observation = env.reset()
    done = False
    navs = []
    while not done:
        action = 1 # stay flat
        observation, reward, done, info = env.step(action)
        navs.append(info['nav'])
        if done:
            print(f'Annualized return: {navs[len(navs)-1]-1}')
            pd.DataFrame(navs).plot()

    #import trading_env as te

    stayflat     = lambda o,e: 1   # stand pat
    buyandhold   = lambda o,e: 2   # buy on day #1 and hold
    randomtrader = lambda o,e: e.action_space.sample() # retail trader

    # to run singly, we call run_strat.  we are returned a dataframe containing 
    #  all steps in the sim.
    bhdf = env.run_strat(buyandhold)

    print(bhdf.head())

    # we can easily plot our nav in time:
    bhdf.bod_nav.plot(title='buy & hold nav')

    env.run_strat(buyandhold).bod_nav.plot(title='same strat, different results')
    env.run_strat(buyandhold).bod_nav.plot()
    env.run_strat(buyandhold).bod_nav.plot()


    # create the tf session
    tf.compat.v1.reset_default_graph() 
    sess1 = tf.compat.v1.InteractiveSession()
    # create policygradient
    pg = PolicyGradient(sess1, obs_dim=5, num_actions=3, learning_rate=1e-2 )

    # and now let's train it and evaluate its progress.  NB: this could take some time...
    df,sf = pg.train_model( env,episodes=25001, log_freq=100)#, load_model=True)