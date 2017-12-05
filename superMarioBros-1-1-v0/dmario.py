import gym
import random
import numpy as np
import getch

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

env = gym.make('SuperMarioBros-1-1-v0')

LR = 1e-3

run = [0, 0, 0, 1, 0, 0]
jump = [0, 0, 0, 1, 0, 0]
run_jump = [0, 0, 0, 1, 1, 0]
do_nothing = [0, 0, 0, 0, 0, 0]


def initial_population():
	prev_obs = env.reset()
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(1):
		score = 0
		game_memory = []
		prev_obs = []
		while True:
			a = getch.getch()
			if a == '1':
				action = run
			elif a == '2':
				action = jump
			elif a == '3':
				action = run_jump
			elif a == '4':
				action = do_nothing
			# action = random.choice([run, jump, run_jump, do_nothing])
			obs, reward, done, info = env.step(action)
			if len(prev_obs) > 0:
				game_memory.append([prev_obs, action])
			prev_obs = obs
			score += reward
			if done: break

		if score >=0:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == run:
					output = [1, 0, 0, 0]
				elif data[1] == jump:
					output = [0, 1, 0, 0]
				elif data[1] == run_jump:
					output = [0, 0, 1, 0]
				elif data[1] == do_nothing:
					output = [0, 0, 0, 1]

				training_data.append([data[0], output])

		env.reset()
		scores.append(score)

	training_data_save = np.array(training_data)
	np.save('mario_001.npy', training_data_save)

	return training_data

def neural_network_model(input_size):
	network = input_data(shape=(None, input_size, 256, 3), name='input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)	

	network = fully_connected(network, 4, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data, model=False):
	# X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	X = np.array([i[0] for i in training_data])
	y = [i[1] for i in training_data]

	model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='mario_001')
	model.save('mario_001.model')
	return model

def get_bool(promt):
	while True:
		try:
			return {'y': True, 'n': False}[input(promt).lower()]
		except KeyError:
			print('Invalid input.. please enter y/n!')


# train_bool = get_bool("Generate Training Data? (y/n)")
# if train_bool:
	# training_data = initial_population()
# else:
training_data = np.load('mario_001.npy')
print('Training Data Loaded!')


# X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
# model = neural_network_model(input_size = len(X[0]))

# train_bool = get_bool("Use Model? (y/n)")
# if train_bool:
	# model.load('mario_001.model')
	# print('Model Loaded!')
# else:
	# model = train_model(training_data, model)

def asdf():
	i=0
	scores = []
	choices = []
	for each_game in range(1):
		score = 0
		game_memory = []
		prev_obs = []
		env.reset()
		while True:
			action = np.argmax(training_data[i])
			print(i)
			i += 1
			# if len(prev_obs) == 0:
				# action = run
				# choices.append(0)
			# else:
				# action = np.argmax(model.predict([prev_obs])[0])
			choices.append(action)
			if action == 0:
				action = run
			elif action == 1:
				action = jump
			elif action == 2:
				action = run_jump
			elif action == 3:
				action = do_nothing

			new_obs, reward, done, info = env.step(action)
			prev_obs = new_obs
			game_memory.append([new_obs, action])
			score += reward
			if done: break

		scores.append(score)
		print(score)

	print('Average Score: ', np.mean(scores))

asdf()
# again = True
# while(again):
#     again = get_bool("Train Model again? (y/n)")
#     if again:
#         train_model(training_data, model)
#     else:
#         asdf()
#         break
