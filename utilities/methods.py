from utilities.environment.quandl_env_src import QuandlEnvSrc
from utilities.environment.trading_env import TradingEnv
from utilities.rl.ddqn_agent import DDQNAgent
from utilities.utils.checks import check_computer_device
from utilities.utils.checks import track_results, use_cpu
import tensorflow as tf
from time import time
import logging

from dotenv import load_dotenv
load_dotenv()

import os

NASDAQ_API = os.getenv("NASDAQ_API")
trading_days = 252

def routine():
    # use_cpu()
    logging.info("Running the routine")
    check_computer_device()
    
    quandl_env_src = QuandlEnvSrc(days=trading_days)
    obs, done = quandl_env_src._step()
    
    trading_environment = TradingEnv(quandl_env_src)
    trading_environment.timestep_limit=trading_days
    trading_environment.seed(42)

    ### Get Environment Params
    state_dim = trading_environment.observation_space.shape[0]
    num_actions = trading_environment.action_space.n
    
    ## Define hyperparameters
    gamma = .99,  # discount factor
    tau = 100  # target network update frequency

    ### NN Architecture
    architecture = (256, 256)  # units per layer
    learning_rate = 0.0001  # learning rate
    l2_reg = 1e-6  # L2 regularization

    ### Experience Replay
    replay_capacity = int(1e6)
    batch_size = 4096

    ### epsilon-greedy Policy
    epsilon_start = 1.0
    epsilon_end = .01
    epsilon_decay_steps = 250
    epsilon_exponential_decay = .99

    ## Criando nossa DDQN
    ## Para tal, iremos usar tensorflow
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
    
    results = ddqn.training(trading_environment)

    # print(results)
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