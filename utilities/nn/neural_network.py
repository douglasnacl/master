from tensorflow.keras.models import Sequential # Para unir várias camadas
from tensorflow.keras.layers import Dense, Dropout # Para Camadas Fully-connected 
from tensorflow.keras.layers import Activation # Função de Ativação
from tensorflow.keras.optimizers import Adam # Gradient Descent operations
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
class NeuralNetwork:

    def __init__(self, state_dim, num_actions, architecture, learning_rate, l2_reg, trainable=True) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.trainable = trainable
        
        self.l2_reg = l2_reg
    
    def build(self):
        layers = []
        optimizer = Adam(learning_rate=self.learning_rate) # SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True) 
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
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
                      optimizer= optimizer)

        return model
        

    