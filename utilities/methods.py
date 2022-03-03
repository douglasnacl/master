from utilities.rl.deep_q_network import DeepQNetwork
from utilities.io.stock_data_generator import StockDataGenerator
from utilities.nn.neural_network import NeuralNetwork
from utilities.rl.deep_q_network import DeepQNetwork

import logging
from dotenv import load_dotenv
load_dotenv()

NUM_ACTIONS = 3 # [buy, sell, hold]
NUM_OBSERVATIONS = 4 # Deve ser 

def routine():
    logging.info("Running the routine")
    
    input_shape = (None, NUM_OBSERVATIONS)
    output_action = NUM_ACTIONS
    nn = NeuralNetwork()
    
    model = nn.build(input_shape, output_action)
    
    dqn = DeepQNetwork()
   