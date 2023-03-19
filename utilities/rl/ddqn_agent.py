from datetime import datetime
from tabnanny import verbose
from utilities.nn.neural_network import NeuralNetwork
from tensorflow.keras.optimizers import Adam
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
import json
import os 

class DoubleDeepQLearningAgent:

  def __init__(
    self, 
    lr=1e-4, # 0.00005, 
    epochs=1, 
    optimizer='SGD', 
    batch_size=4096, 
    model="", 
    depth=0, 
    comment="" #,
    # should_save_weights=False
  ):      
  
    self.model = model
    self.comment = comment
    self.depth = depth
    self.epochs = epochs
    self.optimizer = optimizer
    
    # O espaço de ações varia de 0 a 3
    # 0 - Segurar: manter posição
    # 1 - Comprar: adquirir o ativo
    # 2 - Vender: vender o ativo
    
    self.action_space = np.array([0, 1, 2])
    self.num_actions = len(self.action_space) 
    self.batch_size = batch_size 

    # Defina o tamanho do estado 
    # 5 indicadores padrão do mercado (OHCL) e indicadores calculados
    self.state_size = 5 + self.depth
    
    # Define o repositório onde se salva os modelos
    self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_ddqn_trader"
    
    # Define a capacidade da memória de repetição do treinamento
    replay_capacity = int(1e6)
    self.experience = deque([], maxlen=replay_capacity) 

    # Define a taxa de aprendizado
    self.learning_rate = lr
    # Define o fator de desconto
    self.gamma = .99

    # Define a arquitetura padrão das camadas ocultas da rede target e, por consequencia, online
    self.architecture = (64,128, 64)
    # Define a taxa de regulização l2
    self.l2_reg = 1e-6

    # Define-se epsilon para exploração (exploration vs exploitation)
    self.epsilon = .1 # valor inicial de epsilon  (10%)
    self.epsilon_start = self.epsilon
    self.epsilon_end = .01 # valor final de epsilon (1%)
    self.epsilon_decay_steps = 250 # quantidade de passos de decaimento (250)
    self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps # (0.1 - 0.01) / 250 = 0.000396
    self.epsilon_exponential_decay = .99  # decaimento exponencial para epsilon (0.99)
    self.epsilon_history = []

    # Para aprendizado são utilizadas duas redes, porém para comparação entre st e st+1 é preciso congelar os pesos
    # self.should_save_weights=should_save_weights
    self.online_network = self.build_model()
    self.target_network = self.build_model(trainable=False)
    # if self.should_save_weights:
    #     try:
    #         self.online_network.load_weights(self._newest_file_in_dir('./weights/'))
    #         path = self._newest_file_in_dir('weights/')
    #         self.epsilon = float(path[len('weights/weight_'):len('weights/weight_')+8])
    #     except:
    #         pass
    self.update_target()

    self.reset()

  def _newest_file_in_dir(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

  def reset(self):
    # Inicializa a função com valores nulos
    self.total_steps = self.train_steps = 0
    self.episodes = self.episode_length = self.train_episodes = 0
    self.steps_per_episode = deque(maxlen=self.env_steps_size)
    self.episode_reward = 0
    self.rewards_history = deque(maxlen=self.env_steps_size)

    self.tau = 100 # frequencia de atualização da rede neural
    self.losses = []
    self.idx = tf.range(self.batch_size) # <tf.Tensor: shape=(4000,), dtype=int32, numpy=array([ 0, 1, 2, ..., 3997, 3998, 3999], dtype=int32)>
    self.train = True
    
  def build_model(self, trainable=True):

    # Cria a rede neural utilizada no treinamento 
    neural_network = NeuralNetwork(
        self.state_size,
        self.action_space, 
        self.architecture, 
        self.learning_rate, 
        self.l2_reg,
        optimizer=self.optimizer,
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
    state = state.reshape(-1, self.state_size)
    
    # Realiza a previsão utilizando a rede online para os valores de Q no estado atual
    q = self.online_network.predict(state, verbose=0) 
    
    # Escolhe aleatorimente um número entre 0 e 1 e caso seja menor ou igual a epsilon, uma ação aleatório é executada
    if np.random.rand() < self.epsilon: 
        return  np.random.choice(self.num_actions), q
    # Escolhe a ação onde Q obtem seu máximo valor
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
      if self.train:
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

    # min_value = min(rewards)
    # max_value = max(rewards)
    # rewards = [(x - min_value) / (max_value - min_value) for x in rewards]


    # Realiza a previsão da rede online com base nos valores de q para o próximo estado
    next_q_values = self.online_network.predict_on_batch(next_states) # Q_online(st+1, at+1)
    # Escolhe a ação com maior valor q
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
    # Valores de q previstos - Q_online (st, at) = targets
    q_values = self.online_network.predict_on_batch(states) # Q(st, at)
    q_values[tuple([self.idx, actions])] = targets # Q(st, at) =  rt + 1 * gamma * Q_alvo(st+1, max_(a_(t+1)) Q_online(st+1, at+1)))

    # Treina o modelo
    loss = self.online_network.train_on_batch(x=states, y=q_values) # Q_online(st, rt * gamma * Q_alvo(st+1, max_(a_(t+1)) Q_online(st+1, at+1))))
    self.losses.append(loss)
    self.writer.add_scalar('data/ddql_loss_per_replay', np.sum(self.losses), self.replay_count)
    self.replay_count += 1

    if self.total_steps % self.tau == 0:
        self.update_target()

    return np.sum(self.losses)

  # Cria tensorboard writer
  def create_writer(self, initial_balance, normalize_value, train_episodes):
    self.replay_count = 0
    self.log_dir = 'runs/'+self.log_name
    self.writer = SummaryWriter(self.log_dir)

    # Create folder to save models
    if not os.path.exists(self.log_dir):
        os.makedirs(self.log_dir)

    self.start_training_log(initial_balance, normalize_value, train_episodes)
        
  def start_training_log(self, initial_balance, normalize_value, train_episodes): 
    # Salva os parâmetros de treinamento no arquivo parameters.json par uso futuro
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
    params = {
        "training_start": current_date,
        "initial_balance": initial_balance,
        "training_episodes": train_episodes,
        "depth": self.depth,
        "lr": self.learning_rate,
        "epochs": self.epochs,
        "batch_size": self.batch_size,
        "normalize_value": normalize_value,
        "model": self.model,
        "comment": self.comment,
        "saving_time": "",
        "agent_name": "Double Deep Q Learning",
    }
    with open(self.log_dir+"/parameters.json", "w") as write_file:
        json.dump(params, write_file, indent=4)

  def end_training_log(self):
    with open(self.log_dir+"/parameters.json", "a+") as params:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        params.write(f"training end: {current_date}\n")
        # params['training end'] = f"{current_date}"

  def save(self, name="ddqn_trader", score="", args=[]):
    # Salva os pesos dos modelos (keras model)
    self.online_network.save_weights(f"{self.log_dir}/{score}_{name}.h5")
    # Atualizar as configurações do arquivo json
    if score != "":
        with open(self.log_dir+"/parameters.json", "r") as json_file:
            params = json.load(json_file)
        params["saving_time"] = datetime.now().strftime('%Y-%m-%d %H:%M')
        params["agent_name"] = f"{score}_{name}.h5"
        with open(self.log_dir+"/parameters.json", "w") as write_file:
            json.dump(params, write_file, indent=4)

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
    agent.online_network.load_weights( os.path.join(f"{folder}", f"{name}.h5"))