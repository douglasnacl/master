from tensorflow.keras.models import Sequential # Para unir várias camadas
from tensorflow.keras.layers import Dense, Dropout # Para Camadas Fully-connected 
from tensorflow.keras.layers import Activation # Função de Ativação
from tensorflow.keras.optimizers import Adam # Gradient Descent operations
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
class NeuralNetwork:

  def __init__(self, state_size, action_space, architecture, learning_rate, l2_reg, optimizer='SGD', trainable=True) -> None:
    self.state_size = state_size # (4096, 18)
    self.action_space = action_space # (0, 1, 2)
    self.num_actions = len(self.action_space) # (3) hold, buy e sell
    self.architecture = architecture # (256, 256)
    self.optimizer = optimizer 
    self.learning_rate = learning_rate
    self.trainable = trainable
    
    self.l2_reg = l2_reg

  def get_optimizer(self):
    if self.optimizer == 'SGD':
      return SGD(learning_rate=self.learning_rate, decay=self.learning_rate / 40, momentum=0.9, nesterov=True)
    elif self.optimizer == 'Adam':
      return  Adam(learning_rate=self.learning_rate)
    else:
      raise ValueError("""
        Apenas são permitidos valores SGD ou Adam
      """)

  def build(self):
    # layers = []
    # optimizer = self.get_optimizer()
    # for i, units in enumerate(self.architecture, 1):
    #   layers.append(
    #     Dense(
    #       units=units,
    #       input_dim=self.state_size if i == 1 else None,
    #       activation='relu',
    #       kernel_regularizer=l2(self.l2_reg),
    #       name=f'Dense_{i}',
    #       trainable=self.trainable
    #     )
    #   )
      
    # layers.append(Dropout(.1))
    # layers.append(
    #   Dense(
    #     units=self.num_actions,
    #     trainable=self.trainable,
    #     name='Output'
    #   )
    # )
    # model = Sequential(layers)
    # model.compile(
    #   loss='mean_squared_error',
    #   optimizer= optimizer
    # )

    layers = []
    n = len(self.architecture)
    for i, units in enumerate(self.architecture, 1):
        layers.append(Dense(units=units,
                            input_dim=self.state_size if i == 1 else None,
                            activation='relu',
                            kernel_regularizer=l2(self.l2_reg),
                            name=f'Dense_{i}',
                            trainable=self.trainable))
    layers.append(Dropout(.1))
    layers.append(Dense(units=self.num_actions,
                        trainable=self.trainable,
                        name='Output'))
    model = Sequential(layers)
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=self.learning_rate))

    return model
        

    