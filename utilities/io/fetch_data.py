import pandas as pd
import bitfinex
import time

def fetch_data(start, end, symbol, interval):
    def get_timeframe_in_seconds(timeframe):
      timeframe_dict = {
          '1h': 3600,
          '4h': 14400,
          '15m': 900,
          '1m': 60
      }
      
      try:
        return  timeframe_dict[timeframe]
      except Exception as e:
        raise e

    api_v2 = bitfinex.bitfinex_v2.api_v2()

    data_points = 1000    

    time_in_miliseconds = get_timeframe_in_seconds(interval) * 1000

    start = time.mktime(start.timetuple()) * 1000
    end = time.mktime(end.timetuple()) * 1000
    step = time_in_miliseconds * data_points
    total_steps = (end-start)/time_in_miliseconds
    print(f"\nIniciando a obtenção dos dados do dia {pd.to_datetime(start, unit='ms')} até o dia {pd.to_datetime(end, unit='ms')} em {int(total_steps)} passos\n")
   
    data = []
    while total_steps > 0:
        # recalculando os passos finais
        if total_steps < data_points: 
            step = total_steps * time_in_miliseconds

        end = start + step
        data += api_v2.candles(symbol=symbol, interval=interval, limit=data_points, start=start, end=end)
        print(f"data Inicial: {pd.to_datetime(start, unit='ms')} - data final: {pd.to_datetime(end, unit='ms')} - passos faltantes: {int(total_steps)}")
        start = start + step
        total_steps -= data_points
        time.sleep(1.5)

    # define as colunas dos dados 
    columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
    data = pd.DataFrame(data, columns=columns)
    # remove duplicatas nos dados 
    data.drop_duplicates(inplace=True)
    # define a unidade do intervalo avaliado para ms
    data['Date'] = pd.to_datetime(data['Date'], unit='ms')
    # define o campo data como o indice dos dados
    data.set_index('Date', inplace=True)
    # ordena os dados conforme a data 
    data.sort_index(inplace=True)
    # salva os dados em um csv
    data.to_csv(f"{symbol}_{interval}_.csv")
    return data