from tensorflow.keras.models import Sequential # Para unir várias camadas
from tensorflow.keras.layers import Dense, Dropout # Para Camadas Fully-connected 
from tensorflow.keras.layers import Activation # Função de Ativação
from tensorflow.keras.optimizers import Adam # Gradient Descent operations
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
class NeuralNetwork:

  def __init__(self, state_size, action_space, architecture, learning_rate, l2_reg, trainable=True) -> None:
    self.state_size = state_size # (4096, 18)
    self.action_space = action_space # (0, 1, 2)
    self.num_actions = len(self.action_space) # (3) hold, buy e sell
    self.architecture = architecture # (256, 256)
    self.learning_rate = learning_rate
    self.trainable = trainable
    
    self.l2_reg = l2_reg

  def build(self):
    layers = []
    optimizer = SGD(learning_rate=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True) # Adam(learning_rate=self.learning_rate) # 
    for i, units in enumerate(self.architecture, 1):
      layers.append(
        Dense(
          units=units,
          input_dim=self.state_size if i == 1 else None,
          activation='relu',
          kernel_regularizer=l2(self.l2_reg),
          name=f'Dense_{i}',
          trainable=self.trainable
        )
      )
      
    layers.append(Dropout(.1))
    layers.append(
      Dense(
        units=self.num_actions,
        trainable=self.trainable,
        name='Output'
      )
    )
    model = Sequential(layers)
    model.compile(
      loss='mean_squared_error',
      optimizer= optimizer
    )

    return model
        

    