
from turtle import shape
from utilities.utils._prices2returns import _prices2returns
from matplotlib.pyplot import step
import pandas as pd
import numpy as np

class TradingSim(object):
    """ Implementa o simulador de negociação principal para single-instrument univ """
    def __init__(self, steps, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # muda a cada passo (step)
        self.step             = 0
        self.actions          = np.zeros(self.steps)
        self.navs             = np.ones(self.steps)
        self.mkt_nav          = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.eod_pos          = np.zeros(self.steps)
        self.costs            = np.zeros(self.steps)
        self.trades           = np.zeros(self.steps)
        self.mkt_returns      = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.mkt_nav.fill(1)
        self.strategy_returns.fill(0)
        self.eod_pos.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.mkt_returns.fill(0)
    
    def _step(self, action, return_):
        '''
        Dada uma ação e retorno do período anterior, calcula custos, navs, etc
        e retorna a recompensa e um resumo da atividade do dia.
        '''
        # Pega a posição, nav e mkt nav para o começo do dia (BOD) | = fim do dia anterior (EOD)
        if self.step == 0:
            bod_pos = 0.0
            bod_nav  = 1.0
            mkt_nav  = 1.0
        else: 
            bod_pos = self.eod_pos[self.step-1]
            bod_nav  = self.navs[self.step-1]
            mkt_nav  = self.mkt_nav[self.step-1]


        self.mkt_returns[self.step] = return_
        self.actions[self.step] = action
        
        self.eod_pos[self.step] = action - 1     
        self.trades[self.step] = self.eod_pos[self.step] - bod_pos
        
        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps 
        self.costs[self.step] = trade_costs_pct +  self.time_cost_bps
        reward = ( (bod_pos * return_) - self.costs[self.step] )
        self.strategy_returns[self.step] = reward
        
        # Se não for o primeiro passo armazena o nav e mkt_nav do passo atual
        if self.step != 0 :
            self.navs[self.step] =  bod_nav * (1 + self.strategy_returns[self.step-1])
            self.mkt_nav[self.step] =  mkt_nav * (1 + self.mkt_returns[self.step-1])
        
        info = { 'reward': reward, 'nav':self.navs[self.step],  'mkt_nav':self.mkt_nav[self.step], 'costs':self.costs[self.step], 'strategy_return': self.strategy_returns[self.step] }

        self.step += 1      
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'bod_nav', 'mkt_nav','mkt_return','sim_return',
                'position','costs', 'trade' ]
        rets = _prices2returns(self.navs)
        #pdb.set_trace()
        df = pd.DataFrame( {'action':     self.actions, # today's action (from agent)
                            'bod_nav':    self.navs,    # (Beginning of Day) BOD Net Asset Value (NAV)
                            'mkt_nav':  self.mkt_nav, 
                            'mkt_return': self.mkt_returns,
                            'sim_return': self.strategy_returns,
                            'position':   self.eod_pos,   # (end of day) EOD position
                            'costs':  self.costs,         # eod costs
                            'trade':  self.trades },      # eod trade
                            columns=cols)
        return df   

