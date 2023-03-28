import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from ta.trend import  PSARIndicator
from ta.momentum import rsi

def min_max_normalization(df):
    df = df.copy()
    columns = df.columns.tolist()
    for column in columns[1:]:
        log_difference = np.log(df[column]) - np.log(df[column].shift(1))
        if log_difference[1:].isnull().any():
            df[column] = df[column] - df[column].shift(1)
        else:
            df[column] = np.log(df[column]) - np.log(df[column].shift(1))
        
        min = df[column].min()
        max = df[column].max()
        df[column] = (df[column] - min) / (max - min)
    return df

def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)

def plot_sample_data(df, metric, title='', xlabel='', ylabel=''):
    fig = plt.figure(figsize=(21,8)) 
    plt.plot(df['Date'], df[metric],'-')
    ax =plt.title(title, fontsize=22)
    ax=plt.xlabel(xlabel, fontsize=12)
    ax=plt.ylabel(ylabel, fontsize=12)
    ax=plt.xticks(rotation=90)
    ax=plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.tight_layout()
    plt.show()

def add_indicators(df):
  # Função responsável pela criação e adição de indicadores ao dataframe

  # Adiciona média móvel simples (SMA - Simple Moving Average)
  # df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
  df["sma7"] = df["Close"].rolling(window=7, min_periods=1).mean()
  df["sma25"] = df["Close"].rolling(window=25, min_periods=1).mean()
  df["sma99"] = df["Close"].rolling(window=99, min_periods=1).mean()
  
  # Adiciona Bandas de Bollinger ( Add Bollinger Bands)
  bb_std = df["Close"].rolling(window=20, min_periods=1).std(ddof=0)
  df['bb_sma'] = df["Close"].rolling(window=20, min_periods=1).mean()
  df['bb_ub'] = df["Close"].rolling(window=20, min_periods=1).mean() + bb_std*2
  df['bb_lb'] = df["Close"].rolling(window=20, min_periods=1).mean() - bb_std*2

  # Adiciona indicador SAR Parabólico (Parabolic SAR - Parabolic Stop and Reverse )
  indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
  print("DF: \n", df)
  print("\nPSAR: \n", df['Volume'])
  df['psar'] = indicator_psar.psar()

  # Índice de Força Relativa (RSI - Relative Strength Index)
  df["RSI"] = rsi(close=df["Close"], window=14, fillna=True)
  
  return df

# convert_time = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')# .strftime('%d/%m/%Y-%H')
# # Testando as técnicas de normalização
# dff = pd.read_csv('./BTCUSD_1h_.csv')
# # dff = min_max_normalization(df)
# dff = dff.dropna()
# dff = dff.sort_values('Date')
# dff['Date'] = list(map(convert_time, dff['Date']))

# plot_sample_data(dff, metric='Close', title='Serie Temporal Bitcoin (US$)', xlabel='Data', ylabel='cotação Bitcoin em dólares')

# df_test = add_indicators(dff)
# df_test