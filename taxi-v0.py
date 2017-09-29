import gym
import numpy as np
import time

# Objective is to PICKUP BLUE and DROPOFF at PURPLE following the shortest route possible
env = gym.make("Taxi-v2")
env.env.s = 114

# Q action value table - remember its actions and their associated rewards
Q = np.zeros([env.observation_space.n, env.action_space.n])

# total accumulated reward for each episode
G = 0

# learning rate - mathematical constant phi.
alpha = 0.618

# consecutive wins required
streaks_req = 100
streaks = 0
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
		action = np.argmax(Q[state])
		# '''The agent starts by choosing an action with the highest Q value for the current state using argmax. Argmax will return the index/action with the highest value for that state. Initially, our Q table will be all zeros. But, after every step, the Q values for state-action pairs will be updated.'''
		state2, reward, done, info = env.step(action)
		# '''We update the state-action pair (St , At) for Q using the reward, and the max Q value for state2 (St+1). This update is done using the action value formula (based upon the Bellman equation) and allows state-action pairs to be updated in a recursive fashion (based on future values).'''
		Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action])
		# '''update our total reward G'''
		G += reward
		# '''update state (St) to be the previous state2 (St+1) so the loop can begin again and the next action can be decided'''
		state = state2
	if G > 0:
		streaks += 1
		score_list.append(G)
	else:
		streaks = 0
		score_list = []
	print('Total Reward: {}'.format(G))
	if display:
		break
	if streaks == streaks_req:
		display = True
		print("Solved after {} episodes!".format(episodes))
		print("Average score of {}".format((np.mean(score_list))))
		# break

# score_list=[]
# for i in range(1):
# 	observation = env.reset()
# 	score=0
# 	for t in range(200):
# 		env.render()
# 		print(observation)
# 		action = env.action_space.sample()
# 		obs, rew, done, info = env.step(action)
# 		score = score + rew
# 		if done:
# 				print(t)
# 				score_list.append(score)
# 				break