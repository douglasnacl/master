from ..utilities.environment.quandl_env_src import QuandlEnvSrc
from ..utilities.environment.trading_sim import TradingSim
from ..utilities.environment.trading_env import TradingEnv
import pandas as pd
import numpy as np
import quandl
import os
import gym

QUANDL_TARGET_NAME = "HKEX/00005"
NASDAQ_API = os.getenv("NASDAQ_API")

# def test_quandl_env_src():

#     MinPercentileDays = 100 
#     auth = NASDAQ_API
#     name = QUANDL_TARGET_NAME
#     scale = True

#     # quandl_env_src = QuandlEnvSrc(days=252)
#     # print("DATAFRAME\n", quandl_env_src.data.reset_index())
    
#     df = quandl.get(name) if auth == '' else quandl.get(name, authtoken=auth)
        
#     # df = Date | Nominal Price | Net Change | Change (%) | 
#     # Bid | Ask | P/E(x) | High | Low | Previous Close | 
#     # Share Volume (000) | Turnover (000) | Lot Size
#     print(f'got data for {name} from quandl...')
#     df = df[~np.isnan(df['Share Volume (000)'])][['Nominal Price','Share Volume (000)']]
#     # Agora vamos calcular os retornos e percentis, quando removemos os valores nulos (nan)
#     df = df[['Nominal Price','Share Volume (000)']]
#     # Para os dias com volume zero, vamos substituir 0 por 1 pois os dias não deveriam ter volume 0
#     df['Share Volume (000)'].replace(0,1,inplace=True) 
#     print('Share Volume (000)', df['Share Volume (000)'])
#     # O retorno o será o percentual dado pelo preço nominal menos o preço nominal anterior dividido preço nominal
#     df['Return'] = (df['Nominal Price']-df['Nominal Price'].shift())/df['Nominal Price'].shift()
#     pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
#     print('NOMINAL PRICE\n', (df['Nominal Price']-df['Nominal Price'].shift()))
#     df['ClosePctl'] = df['Nominal Price'].expanding(MinPercentileDays).apply(pctrank)
#     df['VolumePctl'] = df['Share Volume (000)'].expanding(MinPercentileDays).apply(pctrank)
#     # Dropa os valores nulos por linhas (rows/index)
#     df.dropna(axis=0,inplace=True)
#     # Retorno
#     R = df.Return
#     # Escalona os valores para o dagit aframe
#     if scale:
#         mean_values = df.mean(axis=0)
#         std_values = df.std(axis=0)
#         df = (df - np.array(mean_values))/ np.array(std_values)
#     df['Return'] = R # não queremos que nossos retornos sejam dimensionados
#     print('RETURN\n', df['Return'])
#     # Persiste os resultados em variáveis
#     min_values = df.min(axis=0)
#     max_values = df.max(axis=0)
    
#     print("MIN: ", min_values) 
#     print("MAX: ", max_values)

# def test_quandl_env_src():

#     MinPercentileDays = 100 
#     auth = NASDAQ_API
#     name = QUANDL_TARGET_NAME
#     scale = True

#     # quandl_env_src = QuandlEnvSrc(days=252)
#     # print("DATAFRAME\n", quandl_env_src.data.reset_index())
    
#     df = quandl.get(name) if auth == '' else quandl.get(name, authtoken=auth)
        
#     # df = Date | Nominal Price | Net Change | Change (%) | 
#     # Bid | Ask | P/E(x) | High | Low | Previous Close | 
#     # Share Volume (000) | Turnover (000) | Lot Size
#     print(f'got data for {name} from quandl...')
#     df = df[~np.isnan(df['Share Volume (000)'])][['Nominal Price','Share Volume (000)']]
#     # Agora vamos calcular os retornos e percentis, quando removemos os valores nulos (nan)
#     df = df[['Nominal Price','Share Volume (000)']]
#     # Para os dias com volume zero, vamos substituir 0 por 1 pois os dias não deveriam ter volume 0
#     df['Share Volume (000)'].replace(0,1,inplace=True) 
#     print('Share Volume (000)', df['Share Volume (000)'])
#     # O retorno o será o percentual dado pelo preço nominal menos o preço nominal anterior dividido preço nominal
#     df['Return'] = (df['Nominal Price']-df['Nominal Price'].shift())/df['Nominal Price'].shift()
#     pctrank = lambda x: pd.Series(x).rank(pct=True).iloc[-1]
#     print('NOMINAL PRICE\n', (df['Nominal Price']-df['Nominal Price'].shift()))
#     df['ClosePctl'] = df['Nominal Price'].expanding(MinPercentileDays).apply(pctrank)
#     df['VolumePctl'] = df['Share Volume (000)'].expanding(MinPercentileDays).apply(pctrank)
#     # Dropa os valores nulos por linhas (rows/index)
#     df.dropna(axis=0,inplace=True)
#     # Retorno
#     R = df.Return
#     # Escalona os valores para o dagit aframe
#     if scale:
#         mean_values = df.mean(axis=0)
#         std_values = df.std(axis=0)
#         df = (df - np.array(mean_values))/ np.array(std_values)
#     df['Return'] = R # não queremos que nossos retornos sejam dimensionados
#     print('RETURN\n', df['Return'])
#     # Persiste os resultados em variáveis
#     min_values = df.min(axis=0)
#     max_values = df.max(axis=0)
    
#     print("MIN: ", min_values) 
#     print("MAX: ", max_values)

def test_sim_env():
        #     sim = TradingSim(
        #             steps=252,
        #             trading_cost_bps=1e-3,
        #             time_cost_bps=1e-4)
        quandl_env_src = QuandlEnvSrc(days=252)
        #     trading_environment = TradingEnv(quandl_env_src)
        #     num_actions = trading_environment.action_space.n
        #     action = np.random.randint(num_actions)
        print('')
        action_space = gym.spaces.Discrete(3)
        # print("GYM: ", action_space)
        # print("ACTION: ", action, num_actions)
        # assert action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #     observation, done = quandl_env_src._step()
        #     y_return = observation[2] 
        #     print("Y_RETURN: ", y_return)
        #     reward, info = sim._step(action, y_return)     
        #     print("REWARD: ", reward)
        #     print("INFO: ", info)
        
        observation_space =  gym.spaces.Box(
                quandl_env_src.min_values.to_numpy(),
                quandl_env_src.max_values.to_numpy()
        )
        state_dim = observation_space.shape[0]
        print(action_space)
        print(observation_space)
        print(state_dim)

    