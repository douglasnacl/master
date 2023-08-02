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
        ??? Fator de normalização ???
  """
  def __init__(self, df, df_normalized, initial_balance=1000, render_range=100, display_reward=False, display_indicators=False, deterministic=False):
    random.seed(42)
    # Define o espaço de ações e o tamanho do espaço, assim como parametros personalizados
    self.df = df.reset_index()
    self.df_normalized = df_normalized.reset_index()
    self.df_total_steps = len(self.df)-1
    self.initial_balance = initial_balance
    
    # self.lookback_window_size = lookback_window_size
    self._step = 0
    self._used_indices_into_steps = set()
    self.max_interval_tries = 0
    # Faixa de renderização da visualização
    self.render_range = render_range 
    # Define se a recompensa será visualizada no gráfico
    self.display_reward = display_reward
    # Define se os indicadores serão visualizados no gráfico
    self.display_indicators = display_indicators
    # Uma ordem consiste em instruções a um corretor ou corretora para comprar ou vender um título em nome de um investidor. 
    # Uma ordem é a unidade de negociação fundamental de um mercado de valores mobiliários.
    # Define o janela observada no histórico do mercado OHCL=
    self.trades = deque(maxlen=self.render_range) 

    # Taxa padrão cobrada pela Binance (0.1% )
    self.fees = 0.001 
    # Define as colunas, desconsiderando a 'index' e a 'Date'

    self.daily_returns = np.log(df['Close']/df['Close'].shift(1))
    volatility = self.daily_returns.std()  
    self.current_volatility = volatility
    self.agent_daily_return = []

    self.deterministic = deterministic

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
    self.agent_daily_return = []

    logging.info(f"A quantidade permitida de testes são {int((len(self.df_normalized) - 1)/self.env_steps_size)} - total points: {(len(self.df_normalized) - 1)}")
    
    if self.deterministic:
      logging.info("Realizando a execução em um periodo deterministico!")
      
      if self._step == 0:
        self._step = 0
        self._end_step = self.env_steps_size
        self._used_indices_into_steps = set()
        # Quantidade maxima de treinos

      elif self._end_step + 1 not in self._used_indices_into_steps: 
        self._step = self._end_step
        self._end_step = self._step + self.env_steps_size
      # else:

      elif self._end_step > (len(self.df_normalized) - 1) and self.env_steps_size < (len(self.df_normalized) - 1):
        self._step = 0
        self._end_step = self.env_steps_size
        self._used_indices_into_steps = set()
      
      logging.info(f"O intervalo atual vai de  {self._step} até {self._end_step}")

    else:
      logging.info("Realizando a execução em um periodo estocástico!")
      
      if self._step == 0:
        self._step = random.randint(self.env_steps_size, len(self.df_normalized) - 1 - self.env_steps_size)
        self._end_step = self._step + self.env_steps_size   
        self._used_indices_into_steps = set()

      if self.env_steps_size > 0:
        
        self._step = random.randint(self.env_steps_size, len(self.df_normalized) - 1 - self.env_steps_size)
        self._end_step = self._step + self.env_steps_size

        while (self._step in self._used_indices_into_steps or self._end_step in self._used_indices_into_steps) and self.max_interval_tries <= int((len(self.df_normalized) - 1)/self.env_steps_size): 
          
          self._step = random.randint(self.env_steps_size, len(self.df_normalized) - 1 - self.env_steps_size)
          self._end_step = self._step + self.env_steps_size
          self.max_interval_tries += 1
        
      else:
        raise ValueError("O valor dos passos precisa ser maior que zero")
      
      logging.info(f"O intervalo atual vai de  {self._step} até {self._end_step}")
        
        
      
                   
    used_indices_in_this_run = set(range(self._step, self._end_step))
    self._used_indices_into_steps.update(used_indices_in_this_run)

    
    return self.df.loc[self._step] # state[-1]

  def next_observation(self):

    self._step += 1
    obs = self.df.loc[self._step]
  
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
      daily_return_series = pd.Series(data=self.daily_returns)  # create a series with repeating values of daily_return
      self.current_volatility = daily_return_series.rolling(window=30).std().iloc[-1] # 30-day rolling window

    else:
      self.current_volatility = self.daily_returns.std()

    obs = self.next_observation()
    return obs, reward, done
    
  def negotiate_stocks(self, action, state=None):
    current_price = state['Open']
    date = state['Date'] 
    high = state['High'] 
    low = state['Low'] 
    _type = ''

    # Ação Manter Posição - Não realiza nenhuma operação
    if action == 0: 
      self.stock_bought = 0
      self.stock_sold = 0
      _type = 'hold'
      self.episode_orders += 0

    # Ação Comprar - Antes de realizar a operação checa se o balanço atual é maior que 5% do balanço inicial
    elif action == 1 and self.balance > self.initial_balance*0.05:
      # Compra com 100% do saldo atual (balanço financeiro)
      self.stock_bought = self.balance / current_price
      self.stock_bought *= (1-self.fees) # substract fees
      self.balance -= self.stock_bought * current_price
      self.stock_held += self.stock_bought
      _type = 'buy'
      self.episode_orders += 1

    # Ação Vender - Antes de realizar a operação checa se os ativos em mãos vezes o preço do ativo é maior que 5% do balanço inical
    elif action == 2 and self.stock_held * current_price > self.initial_balance*0.05:
      # Vende todas as ações em mãos
      self.stock_sold = self.stock_held
      # Calcula o montante recebido pela venda do ativo dado as taxas
      self.stock_sold *= (1-self.fees) 
      self.balance += self.stock_sold * current_price
      self.stock_held -= self.stock_sold
      _type = 'sell'
      self.episode_orders += 1

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

    # Calculate daily return
    if self.trades and date == self.trades[-1]['Date']:
        agent_daily_return = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        self.agent_daily_return.append(agent_daily_return)

# Função Get Reward: calcula a recompensa de acordo com o tipo de ação realizada - Compra, Venda ou Manter Posição
# considerando:
# Custos de /usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.htmlTransação: O primeiro passo é calcular os custos de transação. Os custos de transação só são aplicados quando uma posição é alterada e o custo é calculado como uma porcentagem do valor da posição atual. Um custo de transação de 1% é assumido nesta implementação.
# Ganho/Perda Percentual: O ganho/perda percentual é calculado com base nos volumes e preços atuais e anteriores. Se a posição atual for "compra" e a posição anterior for "venda" ou "manter", a variação percentual é calculada como (valor_atual - valor_anterior - custo_transação) / valor_anterior. Se a posição atual for "venda" e a posição anterior for "compra" ou "manter", a variação percentual é calculada como (valor_anterior - valor_atual - custo_transação) / valor_anterior. Se a posição atual for "manter", a variação percentual é definida como 0. Se uma ação inválida for tomada, como comprar após uma ordem de compra, uma recompensa negativa é dada.
# Retorno Ajustado ao Risco: O ganho/perda percentual é então ajustado para o risco. Nesta implementação, um índice de Sharpe de 0,5 é assumido, portanto, o retorno ajustado ao risco é calculado como variação_percentual / 0,5.
# Penalidade de Volatilidade: Uma penalidade é aplicada para negociações realizadas durante períodos de alta volatilidade. Nesta implementação, um limite de volatilidade de 2% é assumido. Se a volatilidade atual estiver acima do limite, uma penalidade de -0,1 é aplicada.
# Custo de Oportunidade: Um custo de oportunidade é aplicado para cada negociação realizada. Nesta implementação, um retorno potencial de 2% é assumido para investimentos alternativos. A penalidade de custo de oportunidade é calculada como -custo_oportunidade * ordens_do_episódio.
# Combinando as Recompensas: A recompensa final é a soma do retorno ajustado ao risco, da penalidade de volatilidade e da penalidade de custo de oportunidade. A recompensa é limitada a -1 para evitar grandes recompensas negativas.
# No geral, esta implementação parece levar em consideração vários fatores que podem afetar a lucratividade, como custos de transação, risco, volatilidade e custo de oportunidade.
  
  def get_reward(self):
    if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
        self.prev_episode_orders = self.episode_orders
        current_position = self.trades[-1]['type']
        prev_position = self.trades[-2]['type']

        current_volume = self.trades[-1]['Volume']
        prev_volume = self.trades[-2]['Volume']

        current_price = self.trades[-1]['current_price']
        prev_price = self.trades[-2]['current_price']

        # Calcula os custos de transação 
        transaction_cost = 0.01  # Define o custo de transação como 1%
        if current_position != prev_position:
            # Aplica o custo de transação apenas quando a posição é alterada
            transaction_cost *= (current_volume * current_price) 

        # Calcula o percentual de ganho/perda baseado no volume e preço atual e anterior
        prev_amount = prev_volume * prev_price
        if current_position == "buy" and self.stock_bought:
            current_amount = current_volume * current_price * (1 - transaction_cost)
            percent_change = (current_amount - prev_amount) / prev_amount
        elif current_position == "sell" and self.stock_sold:
            current_amount = prev_volume * current_price * (1 - transaction_cost)
            percent_change = (current_amount - prev_amount) / prev_amount
        elif current_position == "hold":
            current_amount = current_volume * current_price 
            percent_change = 0
        else:
            # Recompença negativa para ações inválidas
            percent_change = -1 

        # Incorpora o retorno ajustado ao risco
        sharpe_ratio =  0.5  # Assume um Sharpe ratio de 0.5
        risk_adjusted_return = percent_change / sharpe_ratio

        # Aplica a penalidade para negociações feitas durante períodos de alta volatilidade
        volatility_threshold = 0.02  # Assume a volatility threshold of 2%
        if self.current_volatility > volatility_threshold:
            volatility_penalty = -0.1  # Penalize trades made during high volatility periods
        else:
            volatility_penalty = 0

        # Aplica a penalidade de custo de oportunidade para cada negociação
        opportunity_cost = 0.02  # Assume um retorno potencial de 2% em investimentos alternativos
        opportunity_cost_penalty = -opportunity_cost * self.episode_orders # Aplique o custo de oportunidade para cada negociação

        # Combina as recompenças e penalidades
        reward = risk_adjusted_return + volatility_penalty + opportunity_cost_penalty
        reward = max(reward, -1)  # Limita a recompença negativa a -1 para evitar recompenças muito negativas 
        
        # logging.info("INFO: Position: {} - Reward: {:5f}".format(current_position, reward))
        self.trades[-1]["Reward"] = reward

        return reward
    else:
        return 0
    
  def render(self, visualize = False):
    if visualize:
      img = self.trading_graph.render(self.df.loc[self._step], self.net_worth, self.trades)
      return img