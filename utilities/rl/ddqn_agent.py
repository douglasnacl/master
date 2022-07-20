from tabnanny import verbose
from utilities.nn.neural_network import NeuralNetwork
from utilities.utils.checks import track_results
from collections import deque
from random import sample
import tensorflow as tf
from time import time
import pandas as pd
import numpy as np

class DDQNAgent:
    def __init__(self, state_dim,           # dimensões do estado
                 num_actions,               # número de açoes possíveis
                 learning_rate,             # taxa de aprendizado
                 gamma,                     # fator de desconto
                 epsilon_start,             # valor inicial de epsilon
                 epsilon_end,               # valor final de epsilon
                 epsilon_decay_steps,       # quantidade de passos de decaimento
                 epsilon_exponential_decay, # decaimento exponencial para epsilon
                 replay_capacity,           # capacidade de mecanismo de repetição (replay)
                 architecture,              # arquitetura de rede neural utilizada em target_network e, por consequencia, online_network
                 l2_reg,                    # taxa de regularização l2
                 tau,                       # frequencia de atualização da rede neural
                 batch_size):               # tamanho do lote (4096)

        epsilon_start = 1.0
        epsilon_end = .01
        epsilon_decay_steps = 250
        epsilon_exponential_decay = .99

        self.state_dim = state_dim # num de dimensões do estado, dada por trading_environment.observation_space.shape[0]
        self.num_actions = num_actions # num ações, dada por trading_environment.action_space.n
        
        self.experience = deque([], maxlen=replay_capacity) # estabelece o tamanho do lote para a experiencia de replay do aprendizado
        
        self.learning_rate = learning_rate 
        self.gamma = gamma

        self.architecture = architecture
        self.l2_reg = l2_reg

        # Para aprendizado são utilizadas duas redes, porém para comparação entre st e st+1 é preciso congelar os pesos
        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        # A política $\epsilon$ começa com valor 'epsilon' e decai 'epsilon_decay' por 'epsilon_decay_steps'
        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps # *250
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps # (1 - 0.01) / 250 = 0.00396
        self.epsilon_exponential_decay = epsilon_exponential_decay # 0.99
        self.epsilon_history = []

        # Inicializa a função com valores nulos
        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size) # <tf.Tensor: shape=(4000,), dtype=int32, numpy=array([   0,    1,    2, ..., 3997, 3998, 3999], dtype=int32)>
        self.train = True

    def build_model(self, trainable=True):

        # cria uma instacia de network que será usada no aprendizado
        neural_network = NeuralNetwork(
            self.state_dim, 
            self.num_actions, 
            self.architecture, 
            self.learning_rate, 
            self.l2_reg,
            trainable=trainable)

        model = neural_network.build()
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights()) # atualiza target_network com os pesos da online_network

    def epsilon_greedy_policy(self, state):
        '''
        Função responsavel pela implementação do paradigma Exporation vs Exploitation. 
        Esta função escolhe entre uma ação aleatória ou aquela que maximia a recompensa (Qmax)
        '''
        self.total_steps += 1 # incrementa o contador de passos
        if np.random.rand() <= self.epsilon: # se o numero aleatório for maior ou igual a epsilon, uma ação aleatório é executada
            return np.random.choice(self.num_actions)
        # Prevê o que usando a rede neural, retorna um conjunto de probilidade por ação
        q = self.online_network.predict(state, verbose=0) # caso contrário, a ação tomada será aquela onde Q obtem seu máximo valor
        action = np.argmax(q, axis=1).squeeze()
        return action

    def memorize_transition(self, state, action, reward, state_prime, not_done):
        '''
        Para a experience replay o agente memoriza cada transição de estado, com isso em mãos será possível
        amostrar aleatóriamente um mini-lote durante a fase de treinamento
        '''
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            if self.train: # se estiver em treinamento, então:
                if self.episodes < self.epsilon_decay_steps: # se o episódio for menor que o epsilon_decay_steps (250), então:
                    self.epsilon -= self.epsilon_decay # reduz o valor de epsilon em epsilon_decay
                else: # caso contrário, epsilon x epsilon_exponential_decay (0.99) 
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            # enquanto o treinamento não estiver terminado a recompensa é armazenada no histórico junto com os passos por episódio
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((state, action, reward, state_prime, not_done))

    def experience_replay(self):
        '''
        O experience_replay ocorre tão logo quanto o lote te
        '''
        # O replay da experiencia memorizada acontece tão logo tenhamos um lote para isto
        # print(self.batch_size)
        if self.batch_size > len(self.experience):
            return
        
        # amostra de minibatch da experiência
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch
        
        # preve os próximos valores Q para escolher a melhor ação
        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        # prevê usando a target network
        next_q_values_target = self.target_network.predict_on_batch(next_states)
        
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))
        
        targets = rewards + not_done * self.gamma * target_q_values

        # Valores de q previstos
        q_values = self.online_network.predict_on_batch(states)
        q_values[tuple([self.idx, actions])] = targets

        # Treina o modelo
        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()
    
        
    def training(self, trading_environment):
        
        # Define o número máximo de episodios e o numero maximo de passos por episodio
        total_steps = 0
        max_episodes = 1000
        max_episode_steps =252

        ### Listas que utilizaremos para armazenar as métricas armazenadas
        episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], [] 
        
        start = time() # variável para controle do tempo de processamento
        results = []
        
        for episode in range(1, max_episodes + 1):
            # Inicialização das listas para calculo do episódio
            this_state_actions, this_state_navs, this_state_mkt_navs, this_state_strategy_return = [], [], [], []
            # Reinicia o ambiente
            this_state = trading_environment.reset()

            for episode_step in range(max_episode_steps):
                # Seleciona a melhor ação baseado na politica epsilon greedy
                action = self.epsilon_greedy_policy(this_state.to_numpy().reshape(-1, self.state_dim))
                next_state, reward, done, info = trading_environment.step(action)
                # observation, reward, done, info 
                # info = { 'reward': reward, 'nav':self.navs[self.step],  'mkt_nav':self.mkt_nav[self.step], 'costs':self.costs[self.step], 'strategy_return': self.strat_retrns[self.step] }
                self.memorize_transition(this_state, 
                                        action, 
                                        reward, 
                                        next_state, 
                                        0.0 if done else 1.0)
                
                this_state_actions.append(action)
                this_state_navs.append(info['nav'])
                this_state_mkt_navs.append(info['mkt_nav'])
                this_state_strategy_return.append(info['strategy_return'])

                if self.train:
                    self.experience_replay()
                if done:
                    break
                this_state = next_state


            # Net Asset Value (NAV) 
            # Episodio começa com NAV  de 1 unidade de dinheiro
            # se o NAV cai para 0, o episódio termina e  o agente perde
            # se o NAV atinge 2.0, o agente vence
            nav =  this_state_navs[-1] * (1 + this_state_strategy_return[-1])

            navs.append(nav)

            market_nav = this_state_mkt_navs[-1]
            market_navs.append(market_nav)

            # track difference between agent an market NAV results
            diff = nav - market_nav
            diffs.append(diff)
            
            if episode % 5 == 0:
                track_results(episode,  
                    # show mov. average results for 100 (10) periods
                    np.mean(navs[-100:]), # nav_mean_100
                    np.mean(navs[-10:]), # nav_mean_10
                    np.mean(market_navs[-100:]),  # market_nav_mean_100
                    np.mean(market_navs[-10:]), # market_nav_10
                    # share of agent wins, defined as higher ending nav
                    np.sum([s > 0 for s in diffs[-100:]])/min(len(diffs), 100),  # win_ratio
                    time() - start, 
                    self.epsilon) # total , epsilon
           
            

            if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
                results = pd.DataFrame({
                    'Episode': list(range(1, episode+1)),
                    'Agent': navs,
                    'Market': market_navs,
                    'Difference': diffs
                }).set_index('Episode')
                print(results.tail())
                break

        trading_environment.close()

        results = pd.DataFrame({
            'Episodes': list(range(1, episode+1)),
            'Agent Avg Return (%)': navs,
            'Market Avg Return (%)': market_navs,
            'Difference [NAV_Agent - NAV_Market]': diffs
        }).set_index('Episode')

        results['Strategy Wins (%)'] = (results['Difference [NAV_Agent - NAV_Market]'] > 0).rolling(100).sum()
        results.info()
        results.describe()
        return results