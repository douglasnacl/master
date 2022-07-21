from utilities.environment.quandl_env_src import QuandlEnvSrc
from utilities.environment.trading_env import TradingEnv
from utilities.rl.ddqn_agent import DDQNAgent
from utilities.utils.checks import check_computer_device
from utilities.utils.checks import track_results, use_cpu, generate_file_name
from datetime import datetime
import tensorflow as tf
from time import time
import logging
import os

NASDAQ_API = os.getenv("NASDAQ_API")
trading_days = 252

def routine(save_weights=False):
    # use_cpu()
    logging.info("Running the routine")
    check_computer_device()

    quandl_env_src = QuandlEnvSrc(days=trading_days)
    
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
    replay_capacity = int(1e6) # 6 * 4096# int(1e3) # int(1e6)
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
                 save_weights=save_weights,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size)
    
    results = ddqn.training(trading_environment)
    results.to_csv(generate_file_name(datetime.now()))
    print(results)
