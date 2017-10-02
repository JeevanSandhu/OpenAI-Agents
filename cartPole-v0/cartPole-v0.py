import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import os

LR = 1e-3

# gym.envs.register(
#     id='CartPoleLong-v0',
#     entry_point='gym.envs.classic_control:CartPoleEnv',
#     max_episode_steps=500,
#     reward_threshold=200.0,
# )
# env = gym.make('CartPoleLong-v0')

env = gym.make("CartPole-v0")
# env.tags['wrapper_config.TimeLimit.max_episode_steps'] = 500
env.reset()
goal_steps = 500
score_requirement = 100
initial_games = 1000000


def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
        # done = False
        # while done != True:
            # env.render()
            # choose random action (0 or 1)
            action = random.randrange(0,2)
            # do it!
            observation, reward, done, info = env.step(action)
            
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here. 
        # all we're doing is reinforcing the score, we're not trying 
        # to influence the machine in any way as to HOW that score is 
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 0:
                    output = [1,0]
                elif data[1] == 1:
                    output = [0,1]
                    
                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('cartPole_training_data_001.npy', training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

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

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]
    
    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_test01')
    model.save('cartPole_001.model')
    return model


def get_bool(prompt):
    while True:
        try:
           return {"y":True,"n":False}[input(prompt).lower()]
        except KeyError:
           print("Invalid input please enter y/n!")


train_bool = get_bool("Generate Training Data? (y/n)")
if train_bool:
    training_data = initial_population()
else:
    training_data = np.load('cartPole_training_data_001.npy')
    print('Training Data Loaded!')

# if os.path.exists('cartPole_training_data_001.npy'):
#     training_data = np.load('cartPole_training_data_001.npy')
#     print('Training Data Loaded!')
# else:
#     training_data = initial_population()

X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
model = neural_network_model(input_size = len(X[0]))

train_bool = get_bool("Use Model? (y/n)")
if train_bool:
    model.load('cartPole_001.model')
    print('Model Loaded!')
else:
    model = train_model(training_data, model)

# if os.path.exists('cartPole_001.model.meta'):
#         model.load('cartPole_001.model')
#         print('Model Loaded!')
# else:
#     model = train_model(training_data, model)

def asdf():
    scores = []
    choices = []
    for each_game in range(100):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)
        print(score)

    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(score_requirement)

again = True
while(again):
    again = get_bool("Train Model again? (y/n)")
    if again:
        train_model(training_data, model)
    else:
        asdf()
        break
