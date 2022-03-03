from tensorflow.keras.models import Sequential # Para unir várias camadas
from tensorflow.keras.layers import Dense # Para Camadas Fully-connected 
from tensorflow.keras.layers import Activation # Função de Ativação
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

class NeuralNetwork:

    @staticmethod
    def build(input_shape, classes):
        # Construindo Input Shape
        
        
        # Cria o base da rede neural
        model = Sequential()
        model.add(Dense(16, input_shape=input_shape))
        # model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Activation('relu'))

        model.add(Dense(32))
        model.add(Activation('relu'))

        model.add(Dense(classes))
        model.add(Activation('linear'))

        return model

    