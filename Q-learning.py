import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax( Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)) )
        # a can be 0,1,2 or 3 and represents the possible action values
                
        # Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        # in this game, s is the position identifier from 0 to 15
        # r is the reward
        # d is False or True in case we got the frisbee or fall into a hole
        # _ is the probability of the action
        
        # Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        # this seems to be 0 or 1 in case we got the fresbee
        rAll += r
        s = s1
        if d == True:
            break
    # jList.append(j)
    rList.append(rAll)

print ("Score over time: " +  str(sum(rList)/num_episodes))
print ("Final Q-Table Values")
print (Q)
