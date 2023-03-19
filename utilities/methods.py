from utilities.environment.trading_env import TradingEnv
from utilities.rl.ddqn_agent import DoubleDeepQLearningAgent
from utilities.io.fetch_data import fetch_data
from utilities.utils.checks import check_computer_device
from utilities.utils.checks import track_results, use_cpu, generate_file_name
from utilities.utils.utilities import add_indicators, format_time, min_max_normalization
from datetime import datetime
from collections import deque
import tensorflow as tf
from time import time
import pandas as pd
import numpy as np
import logging
import os

NASDAQ_API = os.getenv("NASDAQ_API")
trading_days = 252

def train_agent(trading_env, agent, visualize=False, train_episodes = 20, max_train_episode_steps=720):
    # Cria o TensorBoard writer
    agent.create_writer(trading_env.initial_balance, trading_env.normalize_value, train_episodes)
    # Define a janela recente para a quantidade de train_episodes de patrimônio líquido
    total_net_worth = deque(maxlen=train_episodes) 
    # Usado para rastrear o melhor patrimônio líquido médio 
    best_average_net_worth = 0
    start = time()

    for episode in range(train_episodes):

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        state = trading_env.reset(env_steps_size = max_train_episode_steps)

        for _ in range(max_train_episode_steps):
            trading_env.render(visualize)

            # Seleciona a melhor ação baseado na politica epsilon greedy
            action, prediction = agent.act(state)
            
            next_state, reward, done = trading_env.step(action)

            agent.memorize_transition(
                state, 
                action, 
                reward, 
                next_state, 
                0.0 if done else 1.0
            )
            
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state
            loss = agent.experience_replay() #states, actions, rewards, predictions, dones, next_states)

        total_net_worth.append(trading_env.net_worth)
        average_net_worth = np.average(total_net_worth)
        average_reward = np.average(rewards)
        
        agent.writer.add_scalar('data/average net_worth', average_net_worth, episode)
        agent.writer.add_scalar('data/episode_orders', trading_env.episode_orders, episode)
        agent.writer.add_scalar('data/rewards', average_reward, episode)
        
        print("episódio: {:<5} - patrimônio liquído {:<7.2f} - patrimônio liquído médio: {:<7.2f} - pedidos do episódio: {} - tempo de execução: {}  "\
              .format(episode, trading_env.net_worth, average_net_worth, trading_env.episode_orders, format_time(time() - start)))
        
        if episode % 5 == 0:
          tf.keras.backend.clear_session()

       
        if episode >= train_episodes - 1:
            if best_average_net_worth < average_net_worth:
                best_average_net_worth = average_net_worth
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average_net_worth), args=[episode, average_net_worth, trading_env.episode_orders, loss]) 
            agent.save()
 
    agent.end_training_log()


# def test_agent(test_df, test_df_nomalized, visualize=True, test_episodes=10, folder="", name="ddqn_trader", comment="", display_reward=False, display_indicators=False):
#     # with open(folder+"/parameters.json", "r") as json_file:
#     #     params = json.load(json_file)
#     # if name == "":
#     #     params["agent_name"] = f"{name}.h5"
#     # name = params["agent_name"][:-9]

#     agent = DoubleDeepQLearningAgent(lr=0.00001, epochs=5, optimizer=SGD, depth=13, batch_size = 4096)
#     train_env = TradingEnv(df=train_df, df_normalized=train_df_nomalized, display_reward=True, display_indicators=True)

#     agent.load(folder, name)
#     average_net_worth = 0
#     average_orders = 0
#     no_profit_episodes = 0
#     for episode in range(test_episodes):
#         state = train_env.reset(env_steps_size = 720)
#         while True:
#             train_env.render(visualize)
#             action, prediction = agent.act(state)
#             state, reward, done = train_env.step(action)
#             if train_env._step == train_env._end_step:
#                 average_net_worth += train_env.net_worth
#                 average_orders += train_env.episode_orders
#                 if train_env.net_worth < train_env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
#                 print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(episode, train_env.net_worth, average_net_worth/(episode+1), train_env.episode_orders))
#                 break
            
#     print("average {} episodes agent net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
#     print("No profit episodes: {}".format(no_profit_episodes))
#     # save test results to test_results.txt file
#     with open("test_results.txt", "a+") as results:
#         current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
#         results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
#         results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
#         results.write(f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')

def routine(save_weights=False, processing_device="GPU", visualize=False):
    # use_cpu()
    logging.info("Running the routine")

    if processing_device != 'GPU':
        print("INFO: Dispositivo escolhido CPU")
        use_cpu()
    else:
        check_computer_device()

    df = pd.read_csv('./BTCUSD_1h_.csv')
    df = df.dropna()
    df = df.sort_values('Date')

    df = add_indicators(df) # insert indicators to df 2021_02_21_17_54_ddqn_trader
    print(df.shape)

    # df = indicators_dataframe(df, threshold=0.5, plot=False) # insert indicators to df 2021_02_18_21_48_ddqn_trader
    depth = len(list(df.columns[1:])) # OHCL + indicators without Date

    df_nomalized = min_max_normalization(df[99:])[1:].dropna()
    df = df[100:].dropna()

    test_window = 720*3 # 3 months

    # split training and testing datasets
    train_df = df[:-test_window] # we leave 100 to have properly calculated indicators
    test_df = df[-test_window:]

    # split training and testing normalized datasets
    train_df_nomalized = df_nomalized[:-test_window] # we leave 100 to have properly calculated indicators
    # test_df_nomalized = df_nomalized[-test_window:]

    print(f"Pontos totais: {len(df)} - Pontos de treinamento: {len(train_df)} - Pontos de teste: {len(test_df)}")
    
    # single processing training
    agent = DoubleDeepQLearningAgent(lr=0.00001, epochs=5, optimizer='SGD', depth=13, batch_size = 4096)
    train_env = TradingEnv(df=train_df, df_normalized=train_df_nomalized, display_reward=True, display_indicators=True)
    train_agent(train_env, agent, visualize=visualize, train_episodes=100, max_train_episode_steps=720) # visualize=True para visualizar animação
    
    # test_agent(test_df, test_df_nomalized, visualize=True, test_episodes=10, folder="/home/douglasnacl/runs/2023_01_22_19_05_ddqn_trader", name="_ddqn_trader", comment="", display_reward=True, display_indicators=True)

    # quandl_env_src = QuandlEnvSrc(days=trading_days)
    
    # trading_environment = TradingEnv(quandl_env_src)
    # trading_environment.timestep_limit=trading_days
    # trading_environment.seed(42)

    # ### Get Environment Params
    # state_dim = trading_environment.observation_space.shape[0]
    # num_actions = trading_environment.action_space.n
    
    # ## Define hyperparameters
    # gamma = .99,  # discount factor
    # tau = 100  # target network update frequency

    # ### NN Architecture
    # architecture = (256, 256)  # units per layer
    # learning_rate = 0.0001  # learning rate
    # l2_reg = 1e-6  # L2 regularization

    # ### Experience Replay
    # replay_capacity = int(1e6) # 6 * 4096# int(1e3) # int(1e6)
    # batch_size = 4096

    # ### epsilon-greedy Policy
    # epsilon_start = 0.10
    # epsilon_end = .01
    # epsilon_decay_steps = 250
    # epsilon_exponential_decay = .99

    # ## Criando nossa DDQN
    # ## Para tal, iremos usar tensorflow
    # tf.keras.backend.clear_session()
    # ddqn = DDQNAgent(state_dim=state_dim,
    #              num_actions=num_actions,
    #              learning_rate=learning_rate,
    #              gamma=gamma,
    #              epsilon_start=epsilon_start,
    #              epsilon_end=epsilon_end,
    #              epsilon_decay_steps=epsilon_decay_steps,
    #              epsilon_exponential_decay=epsilon_exponential_decay,
    #              replay_capacity=replay_capacity,
    #              architecture=architecture,
    #              save_weights=save_weights,
    #              l2_reg=l2_reg,
    #              tau=tau,
    #              batch_size=batch_size)
    
    # results = ddqn.training(trading_environment)
    # results.to_csv(generate_file_name(datetime.now()))
    # print(results)
