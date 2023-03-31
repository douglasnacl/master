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
import json
import os
from dotenv import load_dotenv
import ast

# Load environment variables from .env file
load_dotenv() # pip install python-dotenv

NASDAQ_API = os.environ.get("NASDAQ_API")
stock_name = os.environ.get("STOCK_NAME", "B3SA")
replay_capacity = int(float(os.environ.get('REPLAY_CAPACITY', '1e6')))
gamma = float(os.environ.get('GAMMA', 0.99))
batch_size = int(os.environ.get('BATCH_SIZE', 360 )) # 4096
nn_architecture = os.environ.get('NN_ARCHITECTURE', '(64, 128, 64)')
nn_learning_rate = float(os.environ.get('NN_LEARNING_RATE', 1e-6))
nn_l2_reg = float(os.environ.get('NN_L2_REG', 0.0001))
nn_activation = os.environ.get('NN_ACTIVATION', 'relu')
nn_optimizer = os.environ.get('NN_OPTIMIZER', 'Adam')
nn_tau = float(os.environ.get('NN_TAU', 100))
model = os.environ.get('MODEL', "Double Deep Q-Learning Network")
comment = os.environ.get('COMMENT', "An agent to learn how to negotiate stocks")
train_episodes = int(os.environ.get("TRAIN_EPISODES", 100))
episode_steps = int(os.environ.get("EPISODE_STEPS", 360))

np.random.seed(42)
tf.random.set_seed(42) 

def train_agent(trading_env, agent, visualize=False, train_episodes=100, max_train_episode_steps=360):
    # Cria o TensorBoard writer
    agent.create_writer(trading_env.initial_balance, train_episodes)
    # Define a janela recente para a quantidade de train_episodes de patrimônio líquido
    total_net_worth = deque(maxlen=train_episodes) 
    # Usado para rastrear o melhor patrimônio líquido médio 
    best_average_net_worth = 0
    win_count = 0  # Initialize win count
    start = time()

    for episode in range(train_episodes):

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        state = trading_env.reset(env_steps_size = max_train_episode_steps)
        benchmark_returns = 0 

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

            # Calculate benchmark return for the current step
            if trading_env._step > 1:
                benchmark_returns += trading_env.daily_returns.iloc[trading_env._step] # (trading_env.df.iloc[trading_env._step]['Close'] - trading_env.df.iloc[trading_env._step-1]['Close'])/trading_env.df.iloc[trading_env._step-1]['Close']
            else:
                benchmark_returns = 0

            loss = agent.experience_replay() #states, actions, rewards, predictions, dones, next_states)
            
        net_returns = trading_env.net_worth - trading_env.initial_balance
        if benchmark_returns is not None:
            info_ratio = information_ratio(net_returns, benchmark_returns)
            agent.writer.add_scalar('data/information_ratio', info_ratio, episode)


        total_net_worth.append(trading_env.net_worth)
        average_net_worth = np.average(total_net_worth)
        average_reward = np.average(rewards)
        episode_reward = sum(rewards)
        
        # Check if agent made a profit and increment win count
        if trading_env.net_worth >= trading_env.initial_balance:
            win_count += 1
        win_rate = win_count / (episode + 1)  # Calculate win rate
        
        agent.writer.add_scalar('data/episode_reward', episode_reward, episode)
        agent.writer.add_scalar('data/average_net_worth', average_net_worth, episode)
        agent.writer.add_scalar('data/episode_orders', trading_env.episode_orders, episode)
        agent.writer.add_scalar('data/rewards', average_reward, episode)
        agent.writer.add_scalar('data/win_rate', win_rate, episode) 
      
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


def test_agent(trading_env, agent, test_df, test_df_nomalized, visualize=True, test_episodes=10, folder="", name="ddqn_trader", comment="", display_reward=False, display_indicators=False):
    with open(folder+"/parameters.json", "r") as json_file:
        params = json.load(json_file)
    if name == "":
        params["agent_name"] = f"{name}.h5"
    name = params["agent_name"][:-9]

    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = trading_env.reset(env_steps_size = 720)
        while True:
            trading_env.render(visualize)
            action, prediction = agent.act(state)
            state, reward, done = trading_env.step(action)
            if trading_env._step == trading_env._end_step:
                average_net_worth += trading_env.net_worth
                average_orders += trading_env.episode_orders
                if trading_env.net_worth < trading_env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(episode, trading_env.net_worth, average_net_worth/(episode+1), trading_env.episode_orders))
                trading_env
            
    print("average {} episodes agent net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')


def information_ratio(net_returns, benchmark_returns):
    active_returns = net_returns - benchmark_returns
    active_std = np.std(active_returns)
    if active_std == 0:
        return 0
    else:
        return np.mean(active_returns) / active_std
    
def routine(save_weights=False, processing_device="GPU", visualize=False):
    logging.info("Running the routine")

    if processing_device != 'GPU':
        print("INFO: Dispositivo escolhido CPU")
        use_cpu()
    else:
        tensors_float = check_computer_device()

    logging.info(f"Realizando a leitura do arquivo de dados {stock_name}")
    df = pd.read_csv(f'./assets/ts/{stock_name}.csv').loc[:, ['Date', 'Open','Close','High','Low','Volume']]#.iloc[:, :-2]#['Date', 'Open','Close','High','Low','Volume']
    df['Volume'] = df['Volume']/1000
    
    df = df.dropna()
    df = df.sort_values('Date')

    df = add_indicators(df) # insert indicators to df 2021_02_21_17_54_ddqn_trader
    
    df_nomalized = min_max_normalization(df[99:])[1:]
    df = df[100:].dropna()
    
    test_window = 180*3 # 720*3 # 3 months

    # Separando os dados em dados de treinamento e teste
    train_df = df[:-test_window] # we leave 100 to have properly calculated indicators
    test_df = df[-test_window:]

    # Separando os dados em dados normalizados de treinamento e teste 
    train_df_nomalized = df_nomalized[:-test_window] # we leave 100 to have properly calculated indicators
    test_df_nomalized = df_nomalized[-test_window:]

    logging.info(f"Pontos totais: {len(df)} - Pontos de treinamento: {len(train_df)} - Pontos de teste: {len(test_df)}")
    
    logging.info("Inicialização das variaveis")

    ## Define parâmetros para o ambiente de treinamento
    logging.info(f"""O treinamento será realizado com base em um conjunto de dados com:
                    > {len(train_df)} dados treinamento
                    > e {len(test_df)} dados de testes
                """)
    trading_env = TradingEnv(df=train_df, df_normalized=train_df_nomalized, display_reward=True, display_indicators=True)

    state_size = len(list(df.columns[1:])) # OHCL + Volume + Indicadores

    nn_architecture_ = ast.literal_eval(nn_architecture)

    action_space = np.array([0, 1, 2])

    logging.info(f"""
        A rede neural usada no treinamento possui:
            > {state_size} neurônios na camada de entrada (OCHL + V + Indicators)
            > {nn_architecture_} neurônios nas camadas ocultas
            > {len(action_space)} neurônios na camada de saida
            > Com taxa de aprendizagem {nn_learning_rate}
            > Regularização L2 {nn_l2_reg}
            > e Otimização {nn_optimizer}
    """)
    
    agent = DoubleDeepQLearningAgent(
        action_space=action_space,
        state_size=state_size, 
        replay_capacity=replay_capacity,
        gamma = gamma, # Fator de Desconto
        batch_size = batch_size, 
        nn_architecture=nn_architecture_, # Arquitetura da Rede Neural
        nn_learning_rate=nn_learning_rate, # Taxa de Aprendizagem
        nn_l2_reg=nn_l2_reg, # Regularização L2
        nn_activation=nn_activation, # Função de Ativação
        nn_optimizer=nn_optimizer, # Otimizador da Rede
        nn_tau=nn_tau, # Frequencia de Atualização da Rede Alvo (Target Network)
        tensors_float=tensors_float, # Define se será 32bit ou 16bit dependendo do hardware do treinamento
        model=model,
        comment=comment,
    )
    train_agent(trading_env, agent, visualize=visualize, train_episodes=train_episodes, max_train_episode_steps=episode_steps) # visualize=True para visualizar animação
    
    # test_agent(test_df, test_df_nomalized, visualize=True, test_episodes=10, folder="/home/douglasnacl/runs/2023_01_22_19_05_ddqn_trader", name="_ddqn_trader", comment="", display_reward=True, display_indicators=True)



# Metrics that can be used to evaluate the performance of a reinforcement learning agent, depending on the specific task and objectives of the agent. Some common metrics include:

# Average reward per episode
# Average profit or net worth per episode
# Sharpe ratio (a measure of risk-adjusted return)
# Maximum drawdown (a measure of risk)
# Win rate (percentage of profitable trades)
# Average holding time (how long the agent holds a position)
# Information ratio (a measure of the agent's ability to generate alpha relative to a benchmark)