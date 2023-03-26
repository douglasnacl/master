from utilities.environment.trading_graph import TradingGraph
from collections import deque
import pandas as pd
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
  def __init__(self, df, df_normalized, initial_balance=1000, render_range=100, display_reward=False, display_indicators=False, normalize_value=40000): # lookback_window_size=50, render_range=100, display_reward=False, display_indicators=False, normalize_value=40000):
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
    # self.orders_history = deque(maxlen=self.render_range) # np.zeros(self.steps)
    # Define o janela observada no histórico do mercado OHCL
    # self.market_history = deque(maxlen=self.render_range)
    self.trades = deque(maxlen=self.render_range) 
    # Define o fator de normalização
    self.normalize_value = normalize_value
    # Taxa padrão cobrada pela Binance (0.1% )
    self.fees = 0.001 
    # Define as colunas, desconsiderando a 'index' e a 'Date'
    self.columns = list(self.df_normalized.columns[2:])
    #self.market_returns = self.market_return(self.df)

    # self._test_reward = 0

    self.daily_returns = np.log(df['Close']/df['Close'].shift(1))
    volatility = self.daily_returns.std()  
    self.current_volatility = volatility



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

    self._last_type = None

    return self.df.loc[self._step] # state[-1]

  def next_observation(self):

    self._step += 1
    obs = self.df.loc[self._step]  # np.concatenate((self.orders_history, self.market_history), axis=1)
  
    return obs
  
  def step(self, action):
    
    self.stock_bought = 0
    self.stock_sold = 0

    self.negotiate_stocks(action, state=self.df.loc[self._step])
    
    reward = self.get_reward()

    if (self.net_worth <= self.initial_balance/2) & (self._step < self._end_step):
      done = True
    else:
      done = False
    
    if self._step >= 30:
      # last_price = self.df_normalized.loc[self._step, 'Close']
      # self.daily_returns = np.log(last_price/self.df_normalized.loc[self._step-1, 'Close'])
      # daily_return_series = pd.Series(data=self.daily_returns)  # create a series with repeating values of daily_return
      # self.current_volatility = daily_return_series.rolling(window=30).std().iloc[-1] # 30-day rolling window

      # last_price = self.df_normalized.loc[self._step, 'Close']
      # daily_return = self.daily_returns # .loc[self._step] # np.log(last_price/self.df_normalized.loc[self._step-1, 'Close'])
      daily_return_series = pd.Series(data=self.daily_returns)  # create a series with repeating values of daily_return
      self.current_volatility = daily_return_series.rolling(window=30).std().iloc[-1] # 30-day rolling window

    else:
      # last_prices = self.df_normalized.loc[:self._step, 'Close']
      # daily_return = np.log(last_prices[1:]/last_prices[:-1])
      self.current_volatility = self.daily_returns.std()

    obs = self.next_observation()
    return obs, reward, done
    
  def negotiate_stocks(self, action, state=None):
    current_price = state['Open']
    date = state['Date'] 
    high = state['High'] 
    low = state['Low'] 
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
    
    if _type != '' or _type:
      self._last_type = _type
    # logging.info("INFO: type: %s - last type: %s", _type, self._last_type)

    self.trades.append({
      'Date' : date, 
      'High' : high, 
      'Low' : low, 
      'Volume': self.stock_sold if _type == 'sell' else (self.stock_bought if _type == 'buy' else self.stock_held), 
      'type': _type, 
      'current_price': current_price
    })
    
    self.prev_net_worth = self.net_worth
    self.net_worth = self.balance + self.stock_held * current_price
  
  # Let me explain what's going on in each step.

  # Transaction Costs: The first step is to calculate the transaction costs. Transaction costs are only applied when a position is changed and the cost is calculated as a percentage of the current position's value. A transaction cost of 1% is assumed in this implementation.
  # Percentage Gain/Loss: The percentage gain/loss is calculated based on the current and previous volumes and prices. If the current position is "buy" and the previous position is "sell" or "hold", the percent change is calculated as (current_amount - prev_amount - transaction_cost) / prev_amount. If the current position is "sell" and the previous position is "buy" or "hold", the percent change is calculated as (prev_amount - current_amount - transaction_cost) / prev_amount. If the current position is "hold", the percent change is set to 0. If an invalid action is taken, such as buying after a buy order, a negative reward is given.
  # Risk-Adjusted Return: The percentage gain/loss is then adjusted for risk. In this implementation, a Sharpe ratio of 0.5 is assumed, so the risk-adjusted return is calculated as percent_change / 0.5.
  # Volatility Penalty: A penalty is applied for trades made during high volatility periods. In this implementation, a volatility threshold of 2% is assumed. If the current volatility is above the threshold, a penalty of -0.1 is applied.
  # Opportunity Cost: An opportunity cost is applied for each trade made. In this implementation, a potential return of 2% is assumed for alternative investments. The opportunity cost penalty is calculated as -opportunity_cost * self.episode_orders.
  # Combine Rewards: The final reward is the sum of the risk-adjusted return, the volatility penalty, and the opportunity cost penalty. The reward is capped at -1 to prevent large negative rewards.
  # Overall, this implementation seems to take into account several factors that can affect profitability, such as transaction costs, risk, volatility, and opportunity cost.
  
  def get_reward(self):
    if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
        self.prev_episode_orders = self.episode_orders
        current_position = self.trades[-1]['type']
        prev_position = self.trades[-2]['type']

        current_volume = self.trades[-1]['Volume']
        prev_volume = self.trades[-2]['Volume']

        current_price = self.trades[-1]['current_price']
        prev_price = self.trades[-2]['current_price']

        # Calculate transaction costs
        transaction_cost = 0.01  # 1% transaction cost
        if current_position != prev_position:
            transaction_cost *= (current_volume * current_price)  # Apply transaction cost only when the position is changed

        # Calculate percentage gain/loss
        prev_amount = prev_volume * prev_price
        if current_position == "buy" and self.stock_bought:# (prev_position == "sell" or self._last_type == 'hold'):# self.stock_held > 0: or 
            # print("BUY")
            current_amount = current_volume * current_price * (1 - transaction_cost)
            percent_change = (current_amount - prev_amount) / prev_amount
        elif current_position == "sell" and self.stock_sold: # and (prev_position == "buy" or self._last_type == 'hold'):
            # print("SELL")
            current_amount = prev_volume * current_price * (1 - transaction_cost)
            percent_change = (current_amount - prev_amount) / prev_amount
        elif current_position == "hold":
            # print("HOLD")
            current_amount = current_volume * current_price 
            percent_change = 0
        else:
            percent_change = -1  # Negative reward for invalid actions

        # Incorporate risk-adjusted return
        sharpe_ratio = 0.5  # Assume a Sharpe ratio of 0.5
        risk_adjusted_return = percent_change / sharpe_ratio

        # Apply a penalty for trades made during high volatility periods
        volatility_threshold = 0.02  # Assume a volatility threshold of 2%
        if self.current_volatility > volatility_threshold:
            volatility_penalty = -0.1  # Penalize trades made during high volatility periods
        else:
            volatility_penalty = 0

        # Apply opportunity cost
        opportunity_cost = 0.02  # Assume a potential return of 2% in alternative investments
        opportunity_cost_penalty = -opportunity_cost * self.episode_orders  # Apply the opportunity cost for each trade

        # Combine the rewards and penalties
        reward = risk_adjusted_return + volatility_penalty + opportunity_cost_penalty
        reward = max(reward, -1)  # Cap the reward at -1 to prevent large negative rewards
        # print("REWARD: ", reward)
        # logging.info("INFO: Position: {} - Reward: {:5f}".format(current_position, reward))
        self.trades[-1]["Reward"] = reward
        # print("TRADE: ", self.trades[-1])

        return reward
    else:
        return 0
    
  def render(self, visualize = False):
    if visualize:
      img = self.trading_graph.render(self.df.loc[self._step], self.net_worth, self.trades)
      return img