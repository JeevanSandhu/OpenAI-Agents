import gym
import numpy as np
import time

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env = gym.make('FrozenLakeNotSlippery-v0')

# Objective is to PICKUP BLUE and DROPOFF at PURPLE following the shortest route possible
# env = gym.make("FrozenLake-v0")

# Q action value table - remember its actions and their associated rewards
Q = np.zeros([env.observation_space.n, env.action_space.n])
# total accumulated reward for each episode
G = 0

# learning rate
learning_rate = 0.81
discount = 0.96

episodes = 0
display = False
score_list = []

while True:
	episodes += 1
	done = False
	G, reward = 0, 0
	state = env.reset()
	while done != True:
		if display:
			time.sleep(1)
			env.render()
    	# '''The agent starts by choosing an action with the highest Q value for the current state using argmax. Argmax will return the index/action with the highest value for that state. Initially, our Q table will be all zeros. But, after every step, the Q values for state-action pairs will be updated.'''
		action = np.argmax(Q[state,:]  + np.random.randn(1, env.action_space.n)*(1./(episodes+1)))
		# '''The agent starts by choosing an action with the highest Q value for the current state using argmax. Argmax will return the index/action with the highest value for that state. Initially, our Q table will be all zeros. But, after every step, the Q values for state-action pairs will be updated.'''
		state2, reward, done, info = env.step(action)
		# '''We update the state-action pair (St , At) for Q using the reward, and the max Q value for state2 (St+1). This update is done using the action value formula (based upon the Bellman equation) and allows state-action pairs to be updated in a recursive fashion (based on future values).'''
		Q[state,action] += learning_rate * (reward + discount*np.max(Q[state2]) - Q[state,action])
		# '''update our total reward G'''
		G += reward
		# '''update state (St) to be the previous state2 (St+1) so the loop can begin again and the next action can be decided'''
		state = state2
	
	score_list.append(G)
	print('Total Reward: {}'.format(G))
	if display:
		break
	if len(score_list)>100:
		if np.mean(score_list[-100:])>0.7:
			display = True
			print("Solved after {} episodes!".format(episodes))
			print("Average score of {}".format((np.mean(score_list[-100:]))))