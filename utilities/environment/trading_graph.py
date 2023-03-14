from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.dates as mpl_dates
import pandas as pd
import numpy as np
import cv2

class TradingGraph:
  _indicators = {}

  _indicators_labels = {
    # Média móvel simples (SMA - Simple Moving Average)
    'sma7' :'7 - Média Móvel Simples',
    'sma25' :'25 - Média Móvel Simples',
    'sma99' :'99 - Média Móvel Simples',
    # Bandas de Bollinger (Add Bollinger Bands)
    'bb_sma' :'Média móvel - Bandas de Bollinger',
    'bb_ub' :'B. superior - Bandas de Bollinger',
    'bb_lb' :'B. inferior - Bandas de Bollinger',
    # SAR Parabólico (Parabolic SAR - Parabolic Stop and Reverse )
    'psar' :'SAR Parabólico',
    # Índice de Força Relativa (RSI - Relative Strength Index)
    'RSI' :'RSI - Índice de Força Relativa'
  }
  action_translation_dict = {
        'buy': 'compra',
        'sell': 'venda',
        'hold': 'segurar'
    }
  # Classe responsável por gerar a visualização utilizando matplotlib utilizada para renderizar os preços da seguinte forma:
  # Métricas: date, open, high, low, close, volume, net_worth, trades
  def __init__(self, render_range, display_reward=False, display_indicators=False):

    self.volume = deque(maxlen=render_range)
    self.net_worth = deque(maxlen=render_range)
    self.render_data = deque(maxlen=render_range)
    self.render_range = render_range
    self.display_reward = display_reward
    self.display_indicators = display_indicators
    # define a aparencia do grafico
    self.define_graph_appearance()
    # define os indicadores caso queiramos mostrá-los
    if self.display_indicators:
      # Cria um novo eixo para os indicadores que compartilham o eixo x (ax2)
      self.ax4 = self.ax2.twinx() # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.twinx.html
      self.instantiate_indicators()

  def define_graph_appearance(self):
    plt.close('all')
    plt.style.use('ggplot')
    # Define o width e height de uma figura  
        
    self.fig = plt.figure(figsize=(16,8)) 

    # Cria o subplot superior para o eixo de preços
    self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    # Cria o subplot inferior para o volume que compartilha o eixo x
    self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)
    # Cria um novo eixo para o patrimonio líquido (net worth) que compartilha o eixo x com o preço
    self.ax3 = self.ax1.twinx()
    self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

  
  def instantiate_indicators(self):
    
    self.sma7, self.sma25, self.sma99, self.bb_sma, self.bb_ub, self.bb_lb, self.psar, self.RSI = \
      [deque(maxlen=self.render_range) for i in range(len(list(self._indicators_labels.keys())))]

    self._indicators = {
        'sma7': self.sma7,
        'sma25': self.sma25,
        'sma99': self.sma99,
        'bb_sma': self.bb_sma,
        'bb_ub': self.bb_ub,
        'bb_lb': self.bb_lb,
        'psar': self.psar,
        'RSI': self.RSI
    }
    self._indicators_linestyle = {
        'sma7': '-',
        'sma25': '-',
        'sma99': '-',
        'bb_sma': '-',
        'bb_ub': '--',
        'bb_lb': '--',
        'psar': '.',
        'RSI': '-'
    }

    self._indicators_colors = {
        'sma7': '#94CCE8',
        'sma25': '#4DB1E3',
        'sma99': '#3C89B0',
        'bb_sma': '#04405E',
        'bb_ub': '#0997DE',
        'bb_lb': '#4DB1E3',
        'psar': '#247096',
        'RSI': '#E36471'
    }

  def update_indicators(self, df):

    for item in list(self._indicators.keys()):
        self._indicators[item].append(df[item])

  def plot_indicators(self, df, date_render_range):

    self.update_indicators(df)

    for item in list(self._indicators_labels.keys())[:-1]:
        self.ax1.plot(date_render_range, self._indicators[item], self._indicators_linestyle[item], color=self._indicators_colors[item], label=self._indicators_labels[item], alpha=0.5)
    self.ax1.legend(loc='upper left')
    
    self.ax4.clear()
    self.ax4.plot(date_render_range, self.RSI, color='green', linestyle=self._indicators_linestyle['RSI'], label=self._indicators_labels['RSI'])
    self.ax4.set_ylabel('RSI') 
    self.ax4.legend(loc='upper left')

  def update_render_range(self):
    # Limpa o eixo da renderização do passo anterior
    self.ax1.clear()
    candlestick_ohlc(self.ax1, self.render_data, width=0.8/24, colorup='#76cf9e', colordown='#fa464f', alpha=0.8)
    
    # Coloca todas as datas em uma lista e preeche o subplot ax2 com o volume
    date_render_range = [i[0] for i in self.render_data]
    self.ax2.clear()
    self.ax2.bar(date_render_range, self.volume, 0.025, label='Volume', alpha=0.5)
    self.ax2.set_ylabel('Volume')
    self.ax2.legend(loc='upper left')
    return date_render_range

  def plot_net_worth(self, date_render_range):
    # Desenha o patrimonio líquido no subplot ax3 (compartilhado com ax1)
    self.ax3.clear()
    self.ax3.plot(date_render_range, self.net_worth, color="blue", label='Patrimônio Líquido', alpha=0.5)
    self.ax3.legend(loc='upper left')

  def translate_trade_type(self, operation_type):
    
    return self.action_translation_dict[operation_type]

  def render(self, df, net_worth, trades):
    # Define variáveis para cada coluna das métricas a serem avaliadas
    date = df["Date"]
    open = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # Adiciona o volume e patrimonio líquido a lista deque
    self.volume.append(volume)
    self.net_worth.append(net_worth)

    # Antes de adicionar a deque realiza a conversão das datas para um formato especial (Matplotlib dates)
    date = mpl_dates.date2num([pd.to_datetime(date)])[0] 

    # Adiciona a data e o OCHL do ativo aos dados de renderização
    self.render_data.append([date, open, high, low, close])

    date_render_range = self.update_render_range()
    
    if self.display_indicators:
        self.plot_indicators(df, date_render_range)

    self.plot_net_worth(date_render_range)
    
    self.ax1.xaxis.set_major_formatter(self.date_format)
    self.fig.autofmt_xdate()

    minimum = np.min(np.array(self.render_data)[:,1:])
    maximum = np.max(np.array(self.render_data)[:,1:])
    RANGE = maximum - minimum
    # venda a descoberto(short sell) e ordens de compras, coloca as setas de pedidos apropriadas
    for trade in trades:
        trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
        if trade_date in date_render_range:
            if trade['type'] == 'buy':
                high_low = trade['Low'] - RANGE*0.02
                ycoords = trade['Low'] - RANGE*0.08
                self.ax1.scatter(trade_date, high_low, c='green', label='compra', s = 120, edgecolors='none', marker="^")
            else:
                high_low = trade['High'] + RANGE*0.02
                ycoords = trade['High'] + RANGE*0.06
                self.ax1.scatter(trade_date, high_low, c='red', label='venda', s = 120, edgecolors='none', marker="v")
            
            if self.display_reward:
                try:
                    self.ax1.annotate(
                        '{0:<5}: {1:.2f}'.format(self.translate_trade_type(trade['type']), trade['Reward']), 
                        (trade_date-0.02, high_low),
                        xytext=(trade_date-0.02, ycoords),
                        bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), 
                        fontsize="small"
                    )
                except:
                    pass

    # Defini-se as layers a cada passo, porque estamos limpando os subplots
    self.ax2.set_xlabel('Data')
    self.ax1.set_ylabel('Preço')
    self.ax3.set_ylabel('Patrimônio Líquido')

    self.fig.tight_layout()

    # Re-desenha o canva
    self.fig.canvas.draw()

    # Converte o canva para imagem
    img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
    
    # img é um rgb, converte para o padrão do opencv (bgr)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Mostra a imagem com opencv
    # cv2_imshow(image) # 
    cv2.imshow("Double Deep Q Lerning trading bot",image)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        return
    else:
        return img