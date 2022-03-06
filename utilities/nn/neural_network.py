from tensorflow.keras.models import Sequential # Para unir várias camadas
from tensorflow.keras.layers import Dense, Dropout # Para Camadas Fully-connected 
from tensorflow.keras.layers import Activation # Função de Ativação
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import clone_model

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
                      optimizer=Adam(learning_rate=self.learning_rate))

        # # Construindo Input Shape
        
        
        # # Cria o base da rede neural
        # model = Sequential()
        # model.add(Dense(16, input_shape=input_shape))
        # # model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        # model.add(Activation('relu'))

        # model.add(Dense(32))
        # model.add(Activation('relu'))

        # model.add(Dense(classes))
        # model.add(Activation('linear'))

        return model

    