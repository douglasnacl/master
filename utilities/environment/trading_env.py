from utilities.environment.trading_graph import TradingGraph
from collections import deque
import numpy as np
import logging
import random

class TradingEnv:
  """
    Class responsável pelo gerenciamento do ambiente de negociação

    Parametros:
      df: dataframe
        Dados
      df_normalized: dataframe
        Dados normalizados
      initial_balance: inta
        Balanço inicial, ou seja, a posição financeira no início do episódio
      lookback_window_size: int
        Janela de observação
      render_range: int
        Faixa de renderização da visualização
      display_reward: boolean
        Mostrar recompensa
      display_indicators: boolean
        Mostrar indicadores
      normalize_value: int
        ??? Fator de normalização ???
  """
  def __init__(self, df, df_normalized, initial_balance=1000, steps = 252, render_range=100, display_reward=False, display_indicators=False, normalize_value=40000): # lookback_window_size=50, render_range=100, display_reward=False, display_indicators=False, normalize_value=40000):
    # Define o espaço de ações e o tamanho do espaço, assim como outros parametros personalizados
    self.df = df.reset_index()
    self.df_normalized = df_normalized.reset_index()
    self.df_total_steps = len(self.df)-1
    self.initial_balance = initial_balance
    
    # self.lookback_window_size = lookback_window_size
    self._step = 0

    # Faixa de renderização da visualização
    self.render_range = render_range 
    # Define se a recompensa será visualizada no gráfico
    self.display_reward = display_reward
    # Define se os indicadores serão visualizados no gráfico
    self.display_indicators = display_indicators
    # Define o janela observada no histórico de ordens
    # Uma ordem consiste em instruções a um corretor ou corretora para comprar ou vender um título em nome de um investidor. 
    # Uma ordem é a unidade de negociação fundamental de um mercado de valores mobiliários.
    # O histórico em questão contém o balanço, patrimonio liquido, cryptos compradas, cryptos vendidas, valores de crypto segurados  
    self.orders_history = deque(maxlen=self.render_range) # np.zeros(self.steps)
    # Define o janela observada no histórico do mercado OHCL
    self.market_history = deque(maxlen=self.render_range)
    self.trades = deque(maxlen=self.render_range) 
    # Define o fator de normalização
    self.normalize_value = normalize_value
    # Taxa padrão cobrada pela Binance (0.1% )
    self.fees = 0.001 
    # Define as colunas, desconsiderando a 'index' e a 'Date'
    self.columns = list(self.df_normalized.columns[2:])
    self.market_returns = self.market_return(self.df)

    self._test_reward = 0

  def reset(self, env_steps_size = 0):
    # Reinicia o estado do ambiente para o estado inicial
    self.trading_graph = TradingGraph(render_range=self.render_range, display_reward=self.display_reward, display_indicators=self.display_indicators) # init visualization
    # Define a janela observada nas negociações

    self.balance = self.initial_balance
    self.net_worth = self.initial_balance
    self.prev_net_worth = self.initial_balance
    self.stock_held = 0
    self.stock_sold = 0
    self.stock_bought = 0
    # Rastreador da contagem de ordens do episódio
    self.episode_orders = 0 
    # Rastreador da contagem de ordens no episódio anterior
    self.prev_episode_orders = 0 
    # Define a janela observada nas recompensas
    self.rewards = deque(maxlen=self.render_range)
    self.env_steps_size = env_steps_size
    self.punish_value = 0
    
    if self.env_steps_size > 0:
      self._step = random.randint(self.env_steps_size, len(self.df_normalized) - 1 - self.env_steps_size)
      self._end_step = self._step + self.env_steps_size
    else:
      self._step = self.env_steps_size
      self._end_step = len(self.df_normalized) - 1

    self.orders_history.append([
          self.balance / self.normalize_value,
          self.net_worth / self.normalize_value,
          self.stock_bought / self.normalize_value,
          self.stock_sold / self.normalize_value,
          self.stock_held / self.normalize_value
        ])
    
    #     # one line for loop to fill market history withing reset call
    self.market_history.append([self.df_normalized.loc[self._step, column] for column in self.columns])
    
    
    self._test_reward = 0

    state = np.concatenate((self.orders_history, self.market_history), axis=1)
    return state[-1]

  def next_observation(self):
    # Função responsável or retornar a próxima observação
    self.market_history.append([self.df_normalized.loc[self._step, column] for column in self.columns])
    obs = np.concatenate((self.orders_history, self.market_history), axis=1)
    return obs[-1]
  
  def step(self, action):
    
    self.stock_bought = 0
    self.stock_sold = 0
    self._step += 1

    self.negotiate_stocks(action)
    
    reward = self.get_reward()
    
    if self.net_worth <= self.initial_balance/2:
      done = True
    else:
      done = False

    if reward != 0:
        self._test_reward += 1
        
        print('Date: ', self.df.loc[self._step, 'Date'],' - Step: ', self._step, ' - High: ', self.df.loc[self._step, 'High'], ' - Low: ', self.df.loc[self._step, 'Low'], ' - total: ', self.stock_held,' - balance: ', self.balance, ' - current_price: ', self.df.loc[self._step, 'Open'])
        print('Reward: ', reward, ' - COUNT: ', self._test_reward, ' - Market Return', self.market_returns[self._step]*self.df.loc[self._step, 'Volume'] )

    
    obs = self.next_observation()
    return obs, reward, done

  def market_return(self, df):
     # calcular o retorno diário
    df['daily_return'] = (df['Close'] - df['Open']) / df['Open']
    
    # calcular o retorno ponderado pelo volume
    df['weighted_return'] = df['daily_return'] * df['Volume']
    
    # calcular o retorno do mercado para cada dia
    return df['weighted_return'].cumsum() / df['Volume'].cumsum()
    
  def negotiate_stocks(self, action):
    current_price = self.df.loc[self._step, 'Open']
    date = self.df.loc[self._step, 'Date'] 
    high = self.df.loc[self._step, 'High'] 
    low = self.df.loc[self._step, 'Low'] 
    _type = ''
    # Segurar (Hold)
    if action == 0: 
      self.stock_bought = 0
      self.stock_sold = 0
      _type = 'hold'
      self.episode_orders += 0
    # Comprar (buy)
    # Antes de realizar a operação checa se o balanço atual é maior que 5% do balanço inicial
    elif action == 1 and self.balance > self.initial_balance*0.05:
      # Compra com 100% do saldo atual (balanço financeiro)
      self.stock_bought = self.balance / current_price
      self.stock_bought *= (1-self.fees) # substract fees
      self.balance -= self.stock_bought * current_price
      self.stock_held += self.stock_bought
      _type = 'buy'
      self.episode_orders += 1

    # Vender (sell)
    # Antes de realizar a operação checa se os ativos em mãos vezes o preço do ativo é maior que 5% do balanço inical
    elif action == 2 and self.stock_held * current_price > self.initial_balance*0.05:
      # Vende 100% das ações seguradas
      self.stock_sold = self.stock_held
      # Calcula o montante recebido pela venda do ativo dado as taxas
      self.stock_sold *= (1-self.fees) 
      self.balance += self.stock_sold * current_price
      self.stock_held -= self.stock_sold
      _type = 'sell'
      self.episode_orders += 1

    self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'total': self.stock_sold, 'type': _type, 'current_price': current_price})
    self.prev_net_worth = self.net_worth
    self.net_worth = self.balance + self.stock_held * current_price

    self.orders_history.append([
      self.balance / self.normalize_value,
      self.net_worth / self.normalize_value,
      self.stock_bought / self.normalize_value,
      self.stock_sold / self.normalize_value,
      self.stock_held / self.normalize_value
    ])

  # Calcula a recompensa
  def get_reward(self):

    if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
      
      self.prev_episode_orders = self.episode_orders
      current_position = self.trades[-1]['type']
      prev_position = self.trades[-2]['type']

      current_volume = self.trades[-1]['total']
      prev_volume = self.trades[-2]['total']

      current_price = self.trades[-1]['current_price']
      prev_price = self.trades[-2]['current_price']
      

      prev_amount = prev_volume * prev_price
      if current_position == "buy" and prev_position == "sell":
        current_amount = prev_volume * current_price
        reward = prev_amount  - current_amount
      
      elif current_position == "sell" and prev_position == "buy": 
        current_amount = current_volume * current_price
        reward =  current_amount - prev_amount

      elif current_position == "hold": 
        current_amount = current_volume * current_price
        reward =  prev_amount - current_amount
      else:
        reward = 0

      self.trades[-1]["Reward"] = reward
      
      return reward

    else:
      return 0

  def render(self, visualize = False):
    if visualize:
      img = self.trading_graph.render(self.df.loc[self._step], self.net_worth, self.trades)
      return img