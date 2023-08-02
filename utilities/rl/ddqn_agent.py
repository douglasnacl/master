from datetime import datetime
from tabnanny import verbose
from utilities.nn.neural_network import NeuralNetwork
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorboardX import SummaryWriter
import tensorflow as tf
from utilities.utils.checks import track_results
from collections import deque
from random import sample
import tensorflow as tf
from time import time
import pandas as pd
import numpy as np
from utilities.utils.checks import generate_file_name_weights, newest_file_in_dir
from utilities.utils.utilities import add_indicators, format_time, min_max_normalization
import json
import os 
import logging

class DoubleDeepQLearningAgent:

  def __init__(
    self, 
    action_space,
    state_size=0, 
    replay_capacity=int(1e6), 
    gamma = .99,
    batch_size=4096, 
    nn_architecture=(64,128,64),
    nn_learning_rate=1e-4, # 0.00005, 
    nn_l2_reg=1e-6,
    nn_activation='relu',
    nn_optimizer='Adam',
    nn_tau=100,
    tensors_float=tf.float32,
    model="", 
    comment="",
  ):      
  
    self.model = model
    self.comment = comment
    self.tensors_float = tensors_float
    self.tf_float = tf.float32 if self.tensors_float == tf.float32 else tf.float16
    self.np_float = np.float32 if self.tensors_float == tf.float32 else np.float16
    
    self.action_space = action_space 
    self.num_actions = len(self.action_space) 
    self.batch_size = batch_size 
  
    # Defina o tamanho do estado 
    # 5 indicadores padrão do mercado (OHCL) e indicadores calculados
    self.state_size = state_size
    
    # Define o repositório onde se salva os modelos
    self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_ddqn_trader"
    
    # Define a capacidade da memória de repetição do treinamento
    replay_capacity = replay_capacity
    self.experience = deque([], maxlen=replay_capacity) 

    # Define a taxa de aprendizado
    self.nn_learning_rate = nn_learning_rate
    self.nn_activation = nn_activation
    # Define o fator de desconto
    self.gamma = gamma

    # Define a arquitetura padrão das camadas ocultas da rede target e, por consequencia, online
    self.nn_architecture = nn_architecture
    # Define a taxa de regulização l2
    self.nn_l2_reg = nn_l2_reg
    self.nn_optimizer = nn_optimizer

    self.nn_tau = nn_tau
    # Define-se epsilon para exploração (exploration vs exploitation)
    self.epsilon = .1 # valor inicial de epsilon  (10%)
    self.epsilon_start = self.epsilon
    self.epsilon_end = .01 # valor final de epsilon (1%)
    self.epsilon_decay_steps = 250 # quantidade de passos de decaimento (250)
    self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps # (0.1 - 0.01) / 250 = 0.000396
    self.epsilon_exponential_decay = .99  # decaimento exponencial para epsilon (0.99)
    self.epsilon_history = []

    # Para aprendizado são utilizadas duas redes, porém para comparação entre st e st+1 é preciso congelar os pesos
    self.online_network = self.build_model()
    self.target_network = self.build_model(trainable=False)

    self.update_target()

    self.reset()

    self.action_0 = 0
    self.action_1 = 0
    self.action_2 = 0
    self.action_random = 0

  def _newest_file_in_dir(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

  def reset(self):
    # Inicializa a função com valores nulos
    self.total_steps = self.train_steps = 0
    self.episodes = self.episode_length = self.train_episodes = 0
    self.steps_per_episode = [] # deque(maxlen=self.env_steps_size)
    self.episode_reward = 0
    self.rewards_history = [] # deque(maxlen=self.env_steps_size)

    self.nn_tau = self.nn_tau # frequencia de atualização da rede neural
    self.losses = []
    self.idx = tf.range(self.batch_size) # <tf.Tensor: shape=(4000,), dtype=int32, numpy=array([ 0, 1, 2, ..., 3997, 3998, 3999], dtype=int32)>
    self.is_training = True
    
  def build_model(self, trainable=True):

    # Cria a rede neural utilizada no treinamento 
    print("Neural Network Architecture: ", self.nn_architecture, " Learning Rate: ", self.nn_learning_rate, " L2 Regularization: ", self.nn_l2_reg, " Optimizer: ", self.nn_optimizer, " Trainable: ", trainable, " State size: ", self.state_size, " Action space: ", self.action_space)
    neural_network = NeuralNetwork(
        self.state_size,
        self.action_space, 
        self.nn_architecture, 
        self.nn_learning_rate, 
        self.nn_l2_reg,
        activation=self.nn_activation,
        optimizer=self.nn_optimizer,
        trainable=trainable)

    model = neural_network.build()
    return model

  def update_target(self):
    # Função responsável por atualizar a rede target com os pesos da rede online
    self.target_network.set_weights(self.online_network.get_weights()) 

  def act(self, state):
    # Função que realiza a ação baseado na política epsilon
    action, prediction = self.epsilon_greedy_policy(state)
    return action, prediction

  def epsilon_greedy_policy(self, state):
    '''
    Função responsavel pela implementação do paradigma Exporation vs Exploitation. 
    Esta função escolhe entre uma ação aleatória ou aquela que maximiza a recompensa (Qmax)
    '''
    # A cada chamada incrementa o contador de passos
    self.total_steps += 1 
    # Realiza o reshape do estado atual para o formato de aceito
    state = np.array(state[2:]).astype(self.np_float)
    state = state.reshape(-1, self.state_size)
    
    # Realiza a previsão utilizando a rede online para os valores de Q no estado atual
    q = self.online_network.predict(state, verbose=0) 
    
    # Escolhe aleatorimente um número entre 0 e 1 e caso seja menor ou igual a epsilon, uma ação aleatório é executada
    if np.random.rand() < self.epsilon: 
      self.action_random += 1
      action, q =  np.random.choice(self.num_actions), q
    # Escolhe a ação onde Q obtem seu máximo valor
    else:
      action = np.argmax(q, axis=1).squeeze()

    return action, q

  def memorize_transition(self, state, action, reward, next_state, not_done):
    '''
    Para a experience replay o agente memoriza cada transição de estado, com isso em mãos será possível
    amostrar aleatóriamente um mini-lote durante a fase de treinamento
    '''
    if not_done:
      self.episode_reward += reward
      self.episode_length += 1
    else:
      # Caso em treinamento, então
      if self.is_training:
        # se o episódio for menor que o epsilon_decay_steps (250), então:
        if self.episodes < self.epsilon_decay_steps: 
          if (self.epsilon - self.epsilon_decay) > self.epsilon_end:
            # Reduz o valor de epsilon em epsilon_decay
            self.epsilon -= self.epsilon_decay 
        # caso contrário, se epsilon x epsilon_exponential_decay (0.99) for maior que epsilon_end, então:
        else: 
          if (self.epsilon * self.epsilon_exponential_decay) > self.epsilon_end:
            self.epsilon *= self.epsilon_exponential_decay

      self.episodes += 1
      self.rewards_history.append(self.episode_reward)
      self.steps_per_episode.append(self.episode_length)
      self.episode_reward, self.episode_length = 0, 0

    # Armazena (st,at,Rt,st+1,done) em uma experience replay
    self.experience.append((state, action, reward, next_state, not_done))

  def experience_replay(self): #, states, actions, rewards, predictions, dones, next_states):
    
    '''
    O experience_replay ocorre tão logo quanto o lote te
    '''
    # A experiencia de repetição ocorre quando a memória de trasição é menor que tamanho o lote definido
    if len(self.experience) < self.batch_size:
      return

    # Empilha as matrizes, formato próprio para o treinamento
    # Obtem uma amostra do minibatch da experiência
    minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
    states, actions, rewards, next_states, not_done = minibatch # et = (st, at, rt, st+1)

    next_states = np.array(next_states[:, 2:]).astype(self.np_float)
    isnan = np.isnan(next_states).any()
    isinf = np.isinf(next_states).any()

    if isnan or isinf:
      ('Input data contains NaN or Inf values')

    scaler = MinMaxScaler(feature_range=(0, 1))
    next_states = scaler.fit_transform(next_states)
    next_states = tf.convert_to_tensor(next_states, dtype=self.tensors_float)
    # Realiza a previsão da rede online com base nos valores de q para o próximo estado
    next_q_values = self.online_network.predict_on_batch(next_states) # Q_online(st+1, at+1)
    # Escolhe a ação com maior valor qZ
    best_actions = tf.argmax(next_q_values, axis=1) #  a*t+1 = max_(a_(t+1)) Q_online(st+1, at+1)
    
    # Realiza a previsão da rede target com base nos valores de q para o próximo estado
    next_q_values_target = self.target_network.predict_on_batch(next_states) # Q_alvo(st+1, at+1)
    
    # Constroi a tabela de valores da rede target
    target_q_values = tf.gather_nd(
      next_q_values_target, # Q_alvo(st+1, at+1))
      tf.stack(
        (
          self.idx,
          tf.cast(best_actions, tf.int32) # max_(a_(t+1)) Q_online(st+1, at+1)
        ), 
        axis=1
      )
    ) #  Q_alvo(st+1, max_(a_(t+1)) Q_online(st+1, at+1)))
    
    # = rt + 1 * gamma * Q_alvo(st+1, max_(a_(t+1)) Q_online(st+1, at+1)))
    targets = rewards + not_done * self.gamma * target_q_values
    states = np.array(states[:, 2:]).astype(self.np_float)
    states = scaler.fit_transform(states)
    states = tf.convert_to_tensor(states, dtype=self.tensors_float)
    # Valores de q previstos - Q_online (st, at) = targets
    q_values = self.online_network.predict_on_batch(states) # Q(st, at)
    q_values[tuple([self.idx, actions])] = targets # Q(st, at) =  rt + 1 * gamma * Q_alvo(st+1, max_(a_(t+1)) Q_online(st+1, at+1)))

    # Treina o modelo
    loss = self.online_network.train_on_batch(x=states, y=q_values) # Q_online(st, rt * gamma * Q_alvo(st+1, max_(a_(t+1)) Q_online(st+1, at+1))))
    
    self.losses.append(loss)
    self.writer.add_scalar('data/ddql_loss_per_replay', np.sum(self.losses), self.replay_count)
    self.replay_count += 1
    if self.total_steps % self.nn_tau == 0:
      self.update_target()
    
    return np.sum(self.losses)

  def information_ratio(self, net_returns, benchmark_returns):
    active_returns = net_returns - benchmark_returns
    active_std = np.std(active_returns)
    if active_std == 0:
        return 0
    else:
        return np.mean(active_returns) / active_std
    
  

  def train(self, trading_env, visualize=False, train_episodes=100, max_train_episode_steps=360):
    # Cria o TensorBoard writer
    
    

    self.create_writer(trading_env.initial_balance, train_episodes)
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
            action, prediction = self.act(state)
            
            next_state, reward, done = trading_env.step(action)

            self.memorize_transition(
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

            loss = self.experience_replay() #states, actions, rewards, predictions, dones, next_states)
            


        agent_daily_return = trading_env.agent_daily_return
        # capm, beta = self.get_capm(trading_env._init_step, trading_env._step, agent_daily_return)

        total_net_worth.append(trading_env.net_worth)
        average_net_worth = np.average(total_net_worth)
        average_reward = np.average(rewards)
        episode_reward = sum(rewards)

        net_returns = average_net_worth - trading_env.initial_balance
        if benchmark_returns is not None:
            info_ratio = self.information_ratio(net_returns, benchmark_returns)
            self.writer.add_scalar('data/information_ratio', info_ratio, episode)

        # Check if agent made a profit and increment win count
        if trading_env.net_worth >= trading_env.initial_balance:
            win_count += 1
        win_rate = win_count / (episode + 1)  # Calculate win rate
        
        self.writer.add_scalar('data/episode_reward', episode_reward, episode)
        self.writer.add_scalar('data/average_net_worth', average_net_worth, episode)
        self.writer.add_scalar('data/perc_average_net_worth', average_net_worth/trading_env.initial_balance - 1, episode)
        self.writer.add_scalar('data/episode_orders', trading_env.episode_orders, episode)
        self.writer.add_scalar('data/rewards', average_reward, episode)
        self.writer.add_scalar('data/win_rate', win_rate, episode) 
        # self.writer.add_scalar('data/capm', capm, episode) 
        # self.writer.add_scalar('data/beta', beta, episode) 
        processing_time = time() - start
        self.writer.add_scalar('data/time_to_process', processing_time, episode) 
      
        print("episódio: {:<5} - patrimônio liquído {:<7.2f} - patrimônio liquído médio: {:<7.2f} - pedidos do episódio: {} - tempo de execução: {}  "\
            .format(episode, trading_env.net_worth, average_net_worth, trading_env.episode_orders, format_time(processing_time)))
        
        if episode % 5 == 0:
          tf.keras.backend.clear_session()

        if episode >= train_episodes - 1:
            if best_average_net_worth < average_net_worth:
                best_average_net_worth = average_net_worth
                print("Saving model")
                self.save(score="{:.2f}".format(best_average_net_worth), args=[episode, average_net_worth, trading_env.episode_orders, loss]) 
            self.save()
 
    self.end_training_log()
  # Cria tensorboard writer
  def create_writer(self, initial_balance, train_episodes):
    self.replay_count = 0
    self.log_dir = 'runs/'+self.log_name
    self.writer = SummaryWriter(self.log_dir)

    # Create folder to save models
    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)

    self.start_training_log(initial_balance, train_episodes)
        
  def start_training_log(self, initial_balance, train_episodes): 
    # Salva os parâmetros de treinamento no arquivo parameters.json par uso futuro
    params = {
      "training_start": datetime.now().strftime('%Y-%m-%d %H:%M'),
      'action_space': str(tuple(self.action_space)),
      "initial_balance": initial_balance,
      "training_episodes": train_episodes,
      "state_size": self.state_size,
      "network_architecture": f"input: {self.state_size} - internals: {self.nn_architecture} - output: {self.action_space}",
      "learning_rate": self.nn_learning_rate,
      "activation": self.nn_activation,
      "optimizer": self.nn_optimizer,
      "batch_size": self.batch_size,
      "model": self.model,
      "comment": self.comment,
      "training_end": ""
    }
    with open(self.log_dir+"/parameters.json", "w") as write_file:
      json.dump(params, write_file, indent=4)
  
  def write_log(self, properties):
    with open(self.log_dir+"/parameters.json", "r") as json_file:
      params = json.load(json_file)
    for property in properties:
      params[property] = properties.get(property)
    with open(self.log_dir+"/parameters.json", "w") as write_file:
      json.dump(params, write_file, indent=4)

  # self.generate_log(properties={'training_end': datetime.now().strftime('%Y-%m-%d %H:%M')})
  def end_training_log(self):
    properties={'training_end': datetime.now().strftime('%Y-%m-%d %H:%M')}
    with open(self.log_dir+"/parameters.json", "a+") as params:
      self.write_log(properties)

  def save(self, name="ddqn_trader", score="", args=[]):
    # Salva os pesos dos modelos (keras model)
    self.online_network.save_weights(f"{self.log_dir}/{score}_{name}.h5")
    # Atualizar as configurações do arquivo json
    if score != "":
      properties={
        'agent_name': f"{score}_{name}.h5"
      }
      self.write_log(properties)

    # Cria log dos argumentos salvos do modelo em um arquivo 
    if len(args) > 0:
      with open(f"{self.log_dir}/log.json", "a+") as log:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        arguments = ""
        for arg in args:
          arguments += f", {arg}"
        log.write(f"{current_time}{arguments}\n")

  def load(self, folder, name):
    # Carrega os pesos do modelo (keras model)
    self.online_network.load_weights( os.path.join(f"{folder}", f"{name}.h5"))