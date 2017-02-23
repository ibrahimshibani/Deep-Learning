import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.initializations import normal
from keras.optimizers import Adam
from keras.models import model_from_json
import pickle
import random
# custom modules
from utils     import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable


NUM_LABELS = 5
ROWS = 30
COLS = 30
CHANNELS_NUM = 4


def append_to_hist(state, obs):
	"""
	Add observation to the state.
	"""
	for i in range(state.shape[0] - 1):
		state[i, :] = state[i + 1, :]
	state[-1, :] = obs

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
						opt.minibatch_size, maxlen)

if opt.disp_on:
	win_all = None
	win_pob = None


# Q Model updated in every iteration
def model_q():
	print("Now we are building the Q-model")
	model = Sequential()
	model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same', input_shape=(ROWS, COLS, CHANNELS_NUM)))
	model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same'))
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='same'))
	model.add(Flatten())
	model.add(Dense(32,activation='relu'))
	model.add(Dense(opt.act_num))

	adam = Adam(lr=1e-6)
	model.compile(loss='mse',optimizer=adam)
	print("We finished building the Q-model")
	return model

	# model = Sequential()
	# model.add(Convolution2D(32, 8, 8, subsample=(4,4), border_mode='same',input_shape=(rows, cols, opt.hist_len)))
	# model.add(Activation('relu'))
	# model.add(Convolution2D(64, 4, 4, subsample=(2,2), border_mode='same'))
	# model.add(Activation('relu'))
	# model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same'))
	# model.add(Activation('relu'))
	# model.add(Flatten())
	# model.add(Dense(32))
	# model.add(Activation('relu'))
	# model.add(Dense(5))

# Target model updated every 10000 iteration
def model_target():
	print("Now we are building the Target model")
	model = Sequential()
	model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same', input_shape=(ROWS, COLS, CHANNELS_NUM)))
	model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same'))
	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='same'))
	model.add(Flatten())
	model.add(Dense(32,activation='relu'))
	model.add(Dense(opt.act_num))
	
	adam = Adam(lr=1e-6)
	model.compile(loss='mse',optimizer=adam)
	print("We finished building the Target-model")
	return model

model_q = model_q()
model_target = model_target()

#Initialize main loop attributes
steps = 1 * 1000010
epi_step = 0
nepisodes = 0
state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
observe = 50000  # Generate random action and do not train for at least 50,000 
explore = 100000 # Simulated annealing constant
INITIAL_EPSILON = 0.75
FINAL_EPSILON = 0.2
epsilon = INITIAL_EPSILON
loss = 0
discount = 0.99
flagt = False # flag for checking if terminal state, used in printing
s = "" #Used for printing Action source, random or network
steps_test = 5000
st = [] #steps - used for graph generation
ll = [] #loss - used for graph generation
re = [] #Rewards
reward_sum = 0
Training = False # Variable used for testing or training
if(Training):
	for step in range(steps):
		if state.terminal or epi_step >= opt.early_stop:
			flagt = True
			epi_step = 0
			nepisodes += 1
			# reset the game
			state = sim.newGame(opt.tgt_y, opt.tgt_x)
			# and reset the history
			state_with_history[:] = 0
			append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
			next_state_with_history = np.copy(state_with_history)
		epi_step += 1
		if random.random() <= epsilon or step < explore:
			action = randrange(opt.act_num)
			s = "Random"
		else:
			reshaped_hist = state_with_history.reshape(1, ROWS, COLS, CHANNELS_NUM)
			action = model_q.predict(reshaped_hist)
			action = np.argmax(action)
			s="Network"

		action_onehot = trans.one_hot_action(action)
		next_state = sim.step(action)
		# append to history
		append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
		# add to the transition table
		trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1),\
			next_state.reward, next_state.terminal)
		# mark next state as current state
		state_with_history = np.copy(next_state_with_history)
		state = next_state

		if step > observe:
			state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
			action_batch_next = model_target.predict_on_batch(next_state_batch.reshape(opt.minibatch_size, ROWS, COLS, CHANNELS_NUM))
			action_batch_next = np.amax(action_batch_next, axis=1)
			action_batch_next = action_batch_next.reshape(opt.minibatch_size, 1)
			updated_Q = model_q.predict_on_batch(state_batch.reshape(opt.minibatch_size, ROWS, COLS, CHANNELS_NUM))

			# Update the Q output ndarray by assingning new value to the highest action
			updated_Q = updated_Q * ((action_batch + 1) % 2) +\
			((1. - terminal_batch) * discount * (action_batch_next) + reward_batch) * action_batch
			
			# Compute model loss
			loss =  model_q.train_on_batch(state_batch.reshape(opt.minibatch_size, ROWS, COLS, CHANNELS_NUM), updated_Q)
			reward_sum += state.reward
			# tprint every 100 steps
			if(step % 100) == 0:
				print("Step    : ", step, " Loss: " ,loss, "Action: ", s, "Terminal: ", flagt )
				print("State action : ", action, " Reward sum : ", reward_sum)
				flagt= False
				st.append(step)
				ll.append(loss)
				re.append(reward_sum)
			# Copy Weights to Target model every 10000 steps
			if (step % 10000) == 0:
				print('Target model\'s weights updated')
				model_target.set_weights(model_q.get_weights())


		# epsilon update rate
		if(epsilon > FINAL_EPSILON and step > observe):
			epsilon -= ((INITIAL_EPSILON - FINAL_EPSILON) / explore)



		# every once in a while you should test your agent here so that you can track its performance
		opt.disp_on = False																																																																																																																				
		if opt.disp_on:
			if win_all is None:
				plt.subplot(121)
				win_all = plt.imshow(state.screen)
				plt.subplot(122)
				win_pob = plt.imshow(state.pob)
			else:
				win_all.set_data(state.screen)
				win_pob.set_data(state.pob)
			plt.pause(opt.disp_interval)
			plt.draw()

		# save weight model at 600k, 800k and 1 million learning iterations to compare how many iterations are needed
		# for the system to converge
		if step == 600000:
			model_json = model_q.to_json()
			with open("fmodel600k.json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model_q.save_weights("fmodel600k.h5")
			print("Saved model-600k to disk")

		if step == 800000:
			model_json = model_q.to_json()
			with open("fmodel800k.json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model_q.save_weights("fmodel800k.h5")
			print("tarSaved model-800k to disk")

		if step == 1000000:
			model_json = model_q.to_json()
			with open("fmodel1000k.json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model_q.save_weights("fmodel1000k.h5")
			print("Saved model-1000k to disk")
			
			# saver.save(sess, dir + '/data-all.chkp')
			with open('steppp', 'wb') as sp:
				pickle.dump(st, sp)
			with open('losss', 'wb') as ls:
				pickle.dump(ll, ls)
			with open('rewarddd', 'wb') as rd:
				pickle.dump(re, rd)
#Used for Testing
else:
	reached = 0
	failed = 0
	json_file = open('fmodel1000k.json','r')
	modell = json_file.read()
	json_file.close()
	ml = model_from_json(modell)
	ml.load_weights("fmodel1000k.h5")
	print("Model loaded")
	state = sim.newGame(opt.tgt_y, opt.tgt_x)
	steps_test = 700
	state_with_history = np.zeros((opt.hist_len, opt.state_siz))
	append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
	next_state_with_history = np.copy(state_with_history)
	for step in range(steps_test):
		if state.terminal or epi_step >= opt.early_stop:
			epi_step = 0
			nepisodes += 1
			if(state.terminal):
				reached += 1
			else:
				failed += 1
			# reset the game
			state = sim.newGame(opt.tgt_y, opt.tgt_x)
			# and reset the history
			state_with_history[:] = 0
			append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
			next_state_with_history = np.copy(state_with_history)
		epi_step += 1
		action = np.argmax(ml.predict((state_with_history).reshape(1,30,30,4)))
		action_onehot = trans.one_hot_action(action)
		#Take next step according to the action selected
		next_state = sim.step(action)
		# append state to history
		append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))

		# mark next state as current state
		state_with_history = np.copy(next_state_with_history)
		state = next_state
		if opt.disp_on:
			if win_all is None:
				plt.subplot(121)
				win_all = plt.imshow(state.screen)
				plt.subplot(122)
				win_pob = plt.imshow(state.pob)
			else:
				win_all.set_data(state.screen)
				win_pob.set_data(state.pob)
			plt.pause(opt.disp_interval)
