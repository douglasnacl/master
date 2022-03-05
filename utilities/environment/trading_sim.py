
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
        self.step         = 0
        self.actions      = np.zeros(self.steps)
        self.navs         = np.ones(self.steps)
        self.mkt_nav      = np.ones(self.steps)
        self.strat_retrns = np.ones(self.steps)
        self.posns        = np.zeros(self.steps)
        self.costs        = np.zeros(self.steps)
        self.trades       = np.zeros(self.steps)
        self.mkt_retrns   = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.mkt_nav.fill(1)
        self.strat_retrns.fill(0)
        self.posns.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.mkt_retrns.fill(0)
    
    def _step(self, action, retrn):
        '''
        Dada uma ação e retorno do período anterior, calcula custos, navs, etc
        e retorna a recompensa e um resumo da atividade do dia.
        '''
        bod_posn = 0.0 if self.step == 0 else self.posns[self.step-1]
        bod_nav  = 1.0 if self.step == 0 else self.navs[self.step-1]
        mkt_nav  = 1.0 if self.step == 0 else self.mkt_nav[self.step-1]

        self.mkt_retrns[self.step] = retrn
        self.actions[self.step] = action
        
        self.posns[self.step] = action - 1     
        self.trades[self.step] = self.posns[self.step] - bod_posn
        
        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps 
        self.costs[self.step] = trade_costs_pct +  self.time_cost_bps
        reward = ( (bod_posn * retrn) - self.costs[self.step] )
        self.strat_retrns[self.step] = reward

        if self.step != 0 :
            self.navs[self.step] =  bod_nav * (1 + self.strat_retrns[self.step-1])
            self.mkt_nav[self.step] =  mkt_nav * (1 + self.mkt_retrns[self.step-1])
        
        info = { 'reward': reward, 'nav':self.navs[self.step], 'costs':self.costs[self.step] }

        self.step += 1      
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'bod_nav', 'mkt_nav','mkt_return','sim_return',
                'position','costs', 'trade' ]
        rets = _prices2returns(self.navs)
        #pdb.set_trace()
        df = pd.DataFrame( {'action':     self.actions, # today's action (from agent)
                            'bod_nav':    self.navs,    # BOD Net Asset Value (NAV)
                            'mkt_nav':  self.mkt_nav, 
                            'mkt_return': self.mkt_retrns,
                            'sim_return': self.strat_retrns,
                            'position':   self.posns,   # EOD position
                            'costs':  self.costs,   # eod costs
                            'trade':  self.trades },# eod trade
                            columns=cols)
        return df   

