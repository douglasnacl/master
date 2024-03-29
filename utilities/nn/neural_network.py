from tensorflow.keras.models import Sequential # Para unir várias camadas
from tensorflow.keras.layers import Dense, Dropout # Para Camadas Fully-connected 
from tensorflow.keras.layers import Activation # Função de Ativação
from tensorflow.keras.optimizers import Adam # Gradient Descent operations
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow import keras
class NeuralNetwork:

  def __init__(self, state_size, action_space, architecture, learning_rate, l2_reg, tensors_float, activation='relu', optimizer='Adam', trainable=True) -> None:
    self.state_size = state_size # (4096, 18)
    self.action_space = action_space # (0, 1, 2)
    self.num_actions = len(self.action_space) # (3) hold, buy e sell
    self.architecture = architecture # (256, 256)
    self.activation = activation
    self.optimizer = optimizer 
    self.learning_rate = learning_rate
    self.trainable = trainable
    self.tensors_float = tensors_float
    self.l2_reg = l2_reg

  def get_optimizer(self):
    if self.optimizer == 'SGD':
      return SGD(learning_rate=self.learning_rate, momentum=0.9)
    elif self.optimizer == 'Adam':
      return  Adam(learning_rate=self.learning_rate)
    else:
      raise ValueError("""
        Apenas são permitidos valores SGD ou Adam
      """)

  def build(self):
    layers = []
    optimizer = self.get_optimizer()

    layers = []
    n = len(self.architecture)
    for i, units in enumerate(self.architecture, 1):
        layers.append(
          Dense(units=units,
              input_dim=self.state_size if i == 1 else None,
              activation=self.activation,
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
        activation='softmax',
        name='Output'
      )
    )
    model = Sequential(layers)
    model.compile(
      optimizer=optimizer,
      loss=self.ddqn_loss
    )

    return model
        
  def ddqn_loss(self, y_true, y_pred):
    # Se y_true tem shape (batch_size, ), realiza reshape para (batch_size, 1)
    if len(y_true.shape) == 1:
        y_true = tf.reshape(y_true, (-1, 1))

    # Calcula os Q-values para as ações tomadas no batch de entrada
    q_values = tf.reduce_sum(y_true * y_pred, axis=1, keepdims=True)

    # Calcula os Q-values para as ações tomadas pela rede alvo
    target_q_values = y_true * (1 - tf.cast(tf.equal(y_pred, tf.reduce_max(y_pred, axis=1, keepdims=True)), dtype=self.tensors_float)) + y_true * tf.cast(tf.equal(y_pred, tf.reduce_max(y_pred, axis=1, keepdims=True)), dtype=self.tensors_float) * q_values

    # Calcula o erro quadrático médio entre os Q-values alvo e os Q-values preditos
    return tf.reduce_mean(tf.square(target_q_values - q_values))