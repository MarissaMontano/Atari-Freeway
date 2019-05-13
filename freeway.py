""" Marissa Montano && Jorge Moreno"""
"""Q-Learning FreeWay"""


import numpy as np
import gym
import matplotlib.pyplot as plt
from scipy import stats
env = gym.make("Freeway-ram-v0")
env.reset()


# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine size of discretized state space
    # Our low number forthis game is 0, therefore we do not need to 0 a low
    num_states = (env.observation_space.high)           
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # Initialize Q table
    Q = np.random.uniform(low = -1, high = 1, size = (num_states[0], num_states[1],env.action_space.n))

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    
    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps)/episodes
    totalRoadsCrossed = []
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        foo = 0
        state = env.reset()
        
        # Discretize state 
        state_adj = (state)
        state_adj = np.round(state_adj, 0).astype(int)
    
        while done != True:   
            """To see the game in action with some episodes please uncoment this code and decide which episodes you want."""
            ## Render environment for last five episodes
            #if i >= (episodes -25):
             #   env.render()
                
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)           
            
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            foo+=reward
        
            #print(state2)
            #How many points it has so far state2[103]
            if (action == 1): # up
                reward += 1 #or 2 

            #if action == 2: 
             #   reward -=2
            #if action == 0: 
             #   reward += 1      
            #print(env.get_action_meanings())

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)
            state2_adj = np.round(state2_adj, 0).astype(int)
  
            #Allow for terminal states
            #state[0] is never >= .5 
           #i think this does if the last step is goal we increment otherwise we would have not done so
            #dont think this is how it works tho

            if done and state[0]>= 0.5:
               # print(done)
                #print(state2[0])
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0], 
                                                   state2_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
            # Update variables
            tot_reward += reward
            state_adj = state2_adj
        #env.render()
        totalRoadsCrossed.append(foo)
       
    # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        # Track rewards Going up getting rewards. Going finish line and getting rewards
        reward_list.append(tot_reward)
        """Important!!!"""
        #change module to represent what episode you want to see average score for
        if (i+1) % 10 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 10 == 0:    
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            
    env.close()
    print(totalRoadsCrossed)
 
    return totalRoadsCrossed
    #ave_reward_list


episode = 10
rewards = QLearning(env, 0.9, 0.3, 0.5, 0, episode)
#alpha learning
#discount = reswards now or later. Short view long view.
#epsilon =  Exploration Rate
#min_eps = min epsilon we want. 0 means no exploration rate

newIntArray = []
for i in range (len(rewards)): 
    newIntArray.append(int(rewards[i]))
max1 = max(newIntArray)
average = sum(newIntArray) / len(newIntArray)

xValue = np.arange(1,(episode+1))
slope, intercept, r_value, p_value, std_err = stats.linregress(xValue,newIntArray)

Regression = slope*xValue+intercept
Formula = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))


plt.plot(xValue, newIntArray, 'o' ,label = 'Data')
plt.plot(xValue, Regression, label ='Regression line: ' + Formula)
plt.plot(max1,label = "Max: " + str(max1))
plt.plot(average, label= "Average: " + str(average))


plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Road Crossed')
plt.title('Number of Chickens Crossed The Road')
plt.savefig('rewardsSample.jpg')     
plt.close() 


