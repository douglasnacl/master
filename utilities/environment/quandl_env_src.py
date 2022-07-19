import pandas as pd
import numpy as np
import logging
import quandl
import os

QUANDL_TARGET_NAME = "HKEX/00005"
NASDAQ_API = os.getenv("NASDAQ_API")

class QuandlEnvSrc(object):
    '''
    Implementação baseada em Quandl de uma fonte de dados da TradingEnv.

    Extrai dados do Quandl, prepara para uso pelo TradingEnv e atua como provedor de dados para cada novo episódio.
    '''
    MinPercentileDays = 100 
    QuandlAuthToken = NASDAQ_API
    Name = QUANDL_TARGET_NAME

    def __init__(self, days=252, name=Name, auth=QuandlAuthToken, scale=True):
        # Carrega as variáveis da entrada
        self.name = name
        self.auth = auth
        self.days = days+1
        logging.info(f'getting data for {QuandlEnvSrc.Name} from quandl...')
        df = quandl.get(self.name) if self.auth == '' else quandl.get(self.name, authtoken=self.auth)
        
        # df = Date | Nominal Price | Net Change | Change (%) | 
        # Bid | Ask | P/E(x) | High | Low | Previous Close | 
        # Share Volume (000) | Turnover (000) | Lot Size

        logging.info(f'got data for {QuandlEnvSrc.Name} from quandl...')
        df = df[~np.isnan(df['Share Volume (000)'])][['Nominal Price','Share Volume (000)']]

        # Agora vamos calcular os retornos e percentis, quando removemos os valores nulos (nan)
        df = df[['Nominal Price','Share Volume (000)']]
        # Para os dias com volume zero, vamos substituir 0 por 1 pois os dias não deveriam ter volume 0
        df['Share Volume (000)'].replace(0,1,inplace=True) 
        # O retorno o será o percentual dado pelo preço nominal menos o preço nominal anterior dividido preço nominal
        df['Return'] = (df['Nominal Price']-df['Nominal Price'].shift())/df['Nominal Price'].shift()
        pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]

        df['ClosePctl'] = df['Nominal Price'].expanding(self.MinPercentileDays).apply(pctrank)
        df['VolumePctl'] = df['Share Volume (000)'].expanding(self.MinPercentileDays).apply(pctrank)
        # Dropa os valores nulos por linhas (rows/index)
        df.dropna(axis=0,inplace=True)
        # Retorno
        R = df.Return
        # Escalona os valores para o dagit aframe
        if scale:
            mean_values = df.mean(axis=0)
            std_values = df.std(axis=0)
            df = (df - np.array(mean_values))/ np.array(std_values)
        df['Return'] = R # não queremos que nossos retornos sejam dimensionados
        # Persiste os resultados em variáveis
        self.min_values = df.min(axis=0)
        self.max_values = df.max(axis=0)
        self.data = df
        self.step = 0
        self.reset()
        self._step()
    
    def reset(self):
        # nós queremos dados contínuos
        self.idx = np.random.randint(low=0, high=len(self.data.index) - self.days)
        self.step = 0
    
    def _step(self):
        # obtem o estado atual. ex: valor da ação
        obs = self.data.iloc[self.idx]
        self.idx += 1
        self.step += 1
        done = self.step >= self.days
        return obs, done

        # [0] Close   [1] Volume   [2]  Return [3] ClosePctl [4] VolumePctl
        # [0] NominalPrice   [1] ShareVolume   [2]  Return [3] ClosePctl [4] VolumePctl

        # SAIDA
        # done:  False
        # obs format:
            # Nominal Price        -0.758694
            # Share Volume (000)    0.217773
            # Return                0.007085
            # ClosePctl            -0.770138
            # VolumePctl            1.018742
            # Name: 2016-06-21 00:00:00, dtype: float64