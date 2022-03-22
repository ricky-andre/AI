# Tutorial written for - Tensorflow 1.15, Keras 2.2.4
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
import pylab
from collections import deque
import tensorflow as tf
import tensorflow.keras as K

def OurModel(input_shape, action_space):
    X_input = K.Input(input_shape)
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = K.layers.Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    # Hidden layer with 256 nodes
    # X = K.layers.Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    # Hidden layer with 64 nodes
    X = K.layers.Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    # Output Layer with # of actions: 2 nodes (left, right)
    X = K.layers.Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    model = K.Model(inputs = X_input, outputs = X)
    model.compile(loss="mse", optimizer=K.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model

class DQNAgent:
    def __init__(self, env_name):
        pylab.figure(figsize=(18, 9))
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(0)  
        # by default, CartPole-v1 has max episode steps = 500
        self.env._max_episode_steps = 2000
        # the shape here is 4
        self.state_size = self.env.observation_space.shape[0]
        # this equals 2, left or right
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        # a sequence container, like an array where elements can also
        # be indexed by their identifier, default value 2000
        self.memory = deque(maxlen=2000)        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 1 - self.epsilon_min
        # 64 is the default value, with 512 the algorithm seems to fail,
        # maybe the buffer gets too big and there's too much past in the algorithm
        self.batch_size = 128
        # raise this to 1million to avoid remembering old states and
        # injecting them into the system, default value 1000
        self.train_start = 1000
        
        # defining double dqn model parameters
        self.ddqn = True
        self.Soft_Update = False
        self.TAU = 0.1 # target network soft update hyperparameter

        self.Save_Path = 'Models'
        self.scores, self.episodes, self.average = [], [], []
        if self.ddqn:
            print("----------Double DQN--------")
            self.Model_name = os.path.join(self.Save_Path,"DDQN_"+self.env_name+".h5")
        else:
            print("-------------DQN------------")
            self.Model_name = os.path.join(self.Save_Path,"DQN_"+self.env_name+".h5")
        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    # after some time interval update the target model to be same with model
    # we distinguish between 'soft updates' and 'hard updates', the soft update
    # considers 90% of the target value and 10% of the old one (percentage is given
    # by the TAU value).
    def update_target_model(self):
        # hard target model update, at the end of every episode we simply copy
        # all the model's weights
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        elif (self.Soft_Update and self.ddqn):
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        # memory is a 'deque' collection object sized 2000, thus
        # when the maximum size is reached, the oldest object is removed
        self.memory.append((state, action, reward, next_state, done))
        # after every step, if the beginning learning period is gone,
        # decrement epsilon as determined by epsilon_decay
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        # epsilon greedy policy, if a random number is below epsilon,
        # choose a random action. In the beginning, epsilon is 1
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        # epsilon is decreased as time passes by, when the training
        # is good, the best action provided by the model is taken
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        # in case the stored experiences are too short, move on
        if len(self.memory) < self.train_start:
            return

        # randomly sample minibatch from the memory, this function 'sample' can take
        # a list, tuple, string, or set and provides back a list of 'K' elements.
        # In the beginning phase, we do not have enough experiences so the number of
        # sampled elements is the minumum between the memory size and the batch size.
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction, for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        # for every element in the minibatch, update Q values
        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            elif self.ddqn:
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a'  Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        # train the Neural Network with batches, here all the complexity is hidden
        # by the NN-model. Fit trains the network by considering the sequence of states
        # that is 'batch_size' long, after every step it calculates the error and than
        # uses gradient descent to change the NN parameters backwards.
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = K.models.load_model(name)

    def save(self, name):
        self.model.save(name)
    
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        # the average here should be calculated on the last X episodes,
        # since the model changes during time improving itself
        if len(self.scores) > 20:
            self.average.append(sum(self.scores[-20:]) / 20)
        else:
            self.average.append(sum(self.scores) / len(self.scores))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        if self.ddqn:
            dqn = 'DDQN_'
        if self.Soft_Update:
            softupdate = '_soft'
        if (episode%10 == 0):
            try:
                pylab.savefig(dqn + self.env_name + softupdate + ".png")
            except OSError:
                pass
        return str(self.average[-1])[:5]
    
    def run(self):
        for epis in range(self.EPISODES):
            state = self.env.reset()
            # this becomes (1, 4)
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # gym function to draw the cartpole in a window
                self.env.render()
                # chosse an action based on the state value of the environment,
                # using epsilon greedy approach
                action = self.act(state)
                # perform an action in the environment, store back what has
                # happened. 
                # next_state (object), reward (float), done (boolean), info
                next_state, reward, done, _ = self.env.step(action)
                # a state is a 4 entries array like the following:
                # [-0.12770472 -0.56449733  0.14814792  1.03449345]
                # coherent with the: self.env.observation_space.shape = (4,)
                # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
                next_state = np.reshape(next_state, [1, self.state_size])
                # to avoid continuing indefinitely, there is an embedded
                # maximum number of steps for each episode
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # every step update target model
                    self.update_target_model()
                    # every episode, plot the result
                    average = self.PlotModel(i, epis)
                    print("episode: {}/{}, score: {}, eps: {:.2}, average: {}".format(epis, self.EPISODES, i, self.epsilon, average))
                    if i == self.env._max_episode_steps:
                        print("Saving trained model as cartpole-ddqn.h5")
                        self.save("cartpole-ddqn.h5")
                        break
                self.replay()

    def test(self):
        self.load("cartpole-ddqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    # the same learning algorithms could be used also for other gym environments,
    # for example pong or pacman
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    agent.run()
    #agent.test()
