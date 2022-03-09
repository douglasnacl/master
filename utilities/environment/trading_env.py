from utilities.environment.trading_sim import TradingSim
from importlib_metadata import metadata
import logging
import gym

class TradingEnv(gym.Env):
    """ Esta classe implementa um ambiente de trading para RL

    O gym disponibiliza observações baseadas nos mercadsos reais de dados obidos
    do Quandl onde, por padrão. Um episódio é definido como _intervalo_ de dias
    contínuos do conjunto de dados geral
    Cada _intervalo_ é um 'passo' no gym e, em cada passo, o algoritmo realiza 
    uma escolha:

    SHORT (0)
    FLAT (1)
    LONG (2)

    Se você negociar, será cobrado, por padrão, 10 BPS do tamanho de sua 
    negociação. Assim, ir de curto para longo custa o dobro do que ir de curto 
    para/do plano. Não negociar também tem um custo padrão de 1 BPS por etapa. 
    Ninguém disse que seria fácil!
    
    No começo do seu episodio, você ira alocar 1 unidade de dinheiro. 
    Este é o seu Valor Patrimonial Líquido (VPL) inicial. Se o seu VPL cai para 
    0, seu episódio acaba e você perde. Se seu VPL atinge 2.0, você ganha.


    O ambiente (env) de negociação acompanhará uma estratégia de compra e retenção que
    atuará como referência para o jogo.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_source_object: object):
        self.days = 252 # self.interval = 252
        self.src = data_source_object
        self.sim = TradingSim(
            steps=self.days,
            trading_cost_bps=1e-3,
            time_cost_bps=1e-4)
        '''
        Ações possíveis
        1 - Buy: Invest all capital for a long position in the stock.
        2 - Flat: Hold cash only
        3 - Sell short: Take a short position equal to the amount of capital.
        '''
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
                self.src.min_values.to_numpy(),
                self.src.max_values.to_numpy()
            )
        self._reset()

    def _reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src._step()[0]

    def _configure(self, display=None):
        self.display = display
    
    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np.random(seed)
        return [seed]
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        observation, done = self.src._step()
        # Close    Volume     Return  ClosePctl  VolumePctl
        yret = observation[2]

        reward, info = self.sim._step(action, yret)
        # info = { 'reward': reward, 'nav':self.navs[self.step],  'mkt_nav':self.mkt_nav[self.step], 'costs':self.costs[self.step], 'strategy_return': self.strat_retrns[self.step] }

        return observation, reward, done, info
    
    def _reset(self):
        self.src.reset()
        self.sim.reset()
        return self.src._step()[0]

    def reset(self):
        return self._reset()
    
    def _render(self, mode='human', close=False):
        pass

    # Algumas funções convenientes
    def run_strat(self, strategy, return_df=True):
        """
        executar a estratégia fornecida, retorna o dataframe com todas as etapas
        """
        observation = self._reset()
        done = False
        while not done:
            action = strategy(observation, self)
            observation, reward, done, info = self.step(action)
        return self.sim.to_df() if return_df else None