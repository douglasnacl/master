from pickletools import optimize
import tensorflow as tf
from tensorflow.keras.models import clone_model
import numpy as np
class DeepQNetwork:
    
    metadata = {'render.modes': ['human']}
    target_model = ''
    
    def __init__(self):
        self.num_actions = 4
        self.num_observations = 4 
        EPOCHS = 1000

        # Hiperparâmetros 
        epsilon = 1.0
        EPSILON_REDUCE = 0.995  # é multiplicado por epsilon a cada época para reduzi-lo
        LEARNING_RATE = 0.001 
        GAMMA = 0.95
        print(f'There are {self.num_actions} possible actions and {self.num_observations} observations')

    # def build(model: tf.keras.engine.sequential.Sequential):
    #     # Deep-Q-Learning works better when using a target network.
    #     target_model = clone_model(model)
    #     pass

    # def load_weight(model: tf.keras.engine.sequential.Sequential, filename: str)
    #     model.load_weights(filename)

    def epsilon_greedy_action_selection(model, epsilon, observation):
        if np.random.random() > epsilon: # Aleatoriamente escolhe-se um numero e caso ele seja maior que epsilon entra no if
            prediction = model.predict(observation) # performa a previsão para a observação
            action = np.argmax(prediction) # Escolhe a ação que tem maior valor
        else:
            action = np.random.randint(0, env.action_space.n) # Caso contrário escolhe uma ação aleatória
        return action

    def replay(replay_buffer, batch_size, model, target_model):

        # As long as the buffer has not enough elements we do nothing
        if len(replay_buffer) < batch_size: 
            return
        
        # Take a random sample from the buffer with size batch_size
        samples = random.sample(replay_buffer, batch_size)  
        
        # to store the targets predicted by the target network for training
        target_batch = []  
        
        # Efficient way to handle the sample by using the zip functionality
        zipped_samples = list(zip(*samples))  
        states, actions, rewards, new_states, dones = zipped_samples  
        
        # Predict targets for all states from the sample
        targets = target_model.predict(np.array(states))
        
        # Predict Q-Values for all new states from the sample
        q_values = model.predict(np.array(new_states))  
        
        # Now we loop over all predicted values to compute the actual targets
        for i in range(batch_size):  
            
            # Take the maximum Q-Value for each sample
            q_value = max(q_values[i][0])  
            
            # Store the ith target in order to update it according to the formula
            target = targets[i].copy()  
            if dones[i]:
                target[0][actions[i]] = rewards[i]
            else:
                target[0][actions[i]] = rewards[i] + q_value * GAMMA
            target_batch.append(target)

        # Fit the model based on the states and the updated targets for 1 epoch
        model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)  

    # def update_model_handler(epoch, update_target_model, model, target_model):
    #     if epoch > 0 and epoch % update_target_model == 0:
    #         target_model.set_weights(model.get_weights())

    def training(self, model):

        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

        best_so_far = 0
        for epoch in range(EPOCHS):
            observation = env.reset()  # Get inital state
            
            # Keras expects the input to be of shape [1, X] thus we have to reshape
            observation = observation.reshape([1, 4])  
            done = False  
            
            points = 0
            while not done:  # as long current run is active
                
                # Select action acc. to strategy
                action = epsilon_greedy_action_selection(model, epsilon, observation)
                
                # Perform action and get next state
                next_observation, reward, done, info = env.step(action)  
                next_observation = next_observation.reshape([1, 4])  # Reshape!!
                replay_buffer.append((observation, action, reward, next_observation, done))  # Update the replay buffer
                observation = next_observation  # update the observation
                points+=1

                # Most important step! Training the model by replaying
                replay(replay_buffer, 32, model, target_model)

            
            epsilon *= EPSILON_REDUCE  # Reduce epsilon
            
            # Check if we need to update the target model
            self.update_model_handler(epoch, update_target_model, model, target_model)
            
            if points > best_so_far:
                best_so_far = points
            if epoch %25 == 0:
                print(f"{epoch}: Points reached: {points} - epsilon: {epsilon} - Best: {best_so_far}")

    # def free_memory():
        
    #     device = cuda.get_current_device()
    #     device.reset()