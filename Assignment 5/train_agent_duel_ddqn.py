import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import Input, Lambda, merge
from keras.layers.convolutional import Convolution2D
from keras.initializations import normal
from keras.optimizers import Adam
from keras.models import model_from_json
from keras import backend as K
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
sim_init = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
						opt.minibatch_size, maxlen)

if opt.disp_on:
	win_all = None
	win_pob = None


# Q Model updated in every iteration
def model_q():
	# print("Now we are building the Q-model")
	# model = Sequential()
	# model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same', input_shape=(ROWS, COLS, CHANNELS_NUM)))
	# model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same'))
	# model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='same'))
	# model.add(Flatten())
	# model.add(Dense(32,activation='relu'))
	# model.add(Dense(opt.act_num))

	input = Input(shape=(ROWS, COLS, CHANNELS_NUM))
	x = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same',
		input_shape=(ROWS, COLS, CHANNELS_NUM),
		trainable=True)(input)
	x = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', trainable=True)(x)
	x = Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu', border_mode='same',  trainable=True)(x)
	x = Flatten()(x)
	# state value tower - V
	state_value = Dense(32, activation='relu', )(x)
	state_value = Dense(1)(state_value)
	state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1), output_shape=(opt.act_num,))(state_value)
	# action advantage tower - A
	action_advantage = Dense(32, activation='relu')(x)
	action_advantage = Dense(opt.act_num)(action_advantage)
	action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(opt.act_num,))(action_advantage)
	# merge to state-action value function Q
	state_action_value = merge([state_value, action_advantage], mode='sum')
	model = Model(input=input, output=state_action_value)
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
	# print("Now we are building the Q-model")
	# model = Sequential()
	# model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same', input_shape=(ROWS, COLS, CHANNELS_NUM)))
	# model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same'))
	# model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', border_mode='same'))
	# model.add(Flatten())
	# model.add(Dense(32,activation='relu'))
	# model.add(Dense(opt.act_num))

	input = Input(shape=(ROWS, COLS, CHANNELS_NUM))
	x = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same',
		input_shape=(ROWS, COLS, CHANNELS_NUM),
		trainable=True)(input)
	x = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', trainable=True)(x)
	x = Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu', border_mode='same', trainable=True)(x)
	x = Flatten()(x)
	# state value tower - V
	state_value = Dense(32, activation='relu')(x)
	state_value = Dense(1)(state_value)
	state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1), output_shape=(opt.act_num,))(state_value)
	# action advantage tower - A
	action_advantage = Dense(32, activation='relu')(x)
	action_advantage = Dense(opt.act_num)(action_advantage)
	action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(opt.act_num,))(action_advantage)
	# merge to state-action value function Q
	state_action_value = merge([state_value, action_advantage], mode='sum')
	model = Model(input=input, output=state_action_value)
	adam = Adam(lr=1e-6)
	model.compile(loss='mse',optimizer=adam)
	print("We finished building the Q-model")
	return model

model_q = model_q()
model_target = model_target()

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
term = []
reward_sum = 0
loss_sum = 0
termcount = 0
Training = False # Variable used for testing or training
json_file = open('train1000k.json','r')
modell = json_file.read()
json_file.close()
ml = model_from_json(modell)
ml.load_weights("train1000k.h5")
print("Model loaded")

def calc_steps(init_x, init_y):
	state = sim_init.newGame_init(init_x, init_y,opt.tgt_y, opt.tgt_x)
	steps_test = 100
	state_with_history = np.zeros((opt.hist_len, opt.state_siz))
	append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
	next_state_with_history = np.copy(state_with_history)
	steps_needed = 0
	epi_step = 0
	nepisodes = 0
	if (init_x == 5 and init_y == 5):
		return 0
	for step in range(steps_test):
		if state.terminal or epi_step >= opt.early_stop:
			epi_step = 0
			nepisodes += 1
			# reset the game
			return steps_needed
			steps_needed = 0
			state = sim_init.newGame_init(init_x, init_y, opt.tgt_y, opt.tgt_x)
			# and reset the history
			state_with_history[:] = 0
			append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
			next_state_with_history = np.copy(state_with_history)
		epi_step += 1
		steps_needed +=1
		action = np.argmax(ml.predict((state_with_history).reshape(1,30,30,4)))
		action_onehot = trans.one_hot_action(action)
		#Take next step according to the action selected
		next_state = sim_init.step2(action)
		# append state to history
		append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))

		# mark next state as current state
		state_with_history = np.copy(next_state_with_history)
		state = next_state
	return 0

erro = []
flagter = False
for step in range(steps):

	if state.terminal or epi_step >= opt.early_stop:
		if (state.terminal == True):
			flagter = True
		flagt = True
		nepisodes += 1
		# reset the game
		ini_x, ini_y = sim.get_init()
		state = sim.newGame(opt.tgt_y, opt.tgt_x)
		optimal_step = calc_steps(ini_x, ini_y)
		error = epi_step - optimal_step
		if (flagter == False):
			error = 100
		flagter = False
		erro.append(error)
		print( "Initial: ", ini_x, ", ", ini_y, " Optimal : ", optimal_step, " Steps took : ", epi_step , " Error of Episode : ", error)
		epi_step = 0
		# and reset the history
		state_with_history[:] = 0
		append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
		next_state_with_history = np.copy(state_with_history)
		ll.append(loss_sum)
		loss_sum = 0
		termcount += 1
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
		# action_batch_next = np.amax(action_batch_next, axis=1)
		# action_batch_next = action_batch_next.reshape(opt.minibatch_size, 1)
		updated_Q = model_q.predict_on_batch(state_batch.reshape(opt.minibatch_size, ROWS, COLS, CHANNELS_NUM))

		# Update the Q output ndarray by assingning new value to the highest action
		# updated_Q = updated_Q * ((action_batch + 1) % 2) +\
		# ((1. - terminal_batch) * discount * (action_batch_next) + reward_batch) * action_batch

		best_action = np.argmax(updated_Q, axis=1)

		# DDQN
		for i in range(0,opt.minibatch_size):
			updated_Q[i,np.argmax(action_batch[i])] =  reward_batch[i] + ((1-terminal_batch[i]) * discount * action_batch_next[i][best_action[i]])

		loss =  model_q.train_on_batch(state_batch.reshape(opt.minibatch_size, ROWS, COLS, CHANNELS_NUM), updated_Q)
		reward_sum += state.reward
		loss_sum += loss
		st.append(step)
		re.append(reward_sum)
		term.append(termcount)			
		# tprint every 100 steps
		if(step % 1000) == 0:
			print("Step    : ", step, " Loss: " ,loss, "Action: ", s, "Terminal: ", flagt )
			print("State action : ", action, " Reward sum : ", reward_sum)
			flagt= False
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
	if step == 700000:
		model_json = model_q.to_json()
		with open("duel_dd_error600k.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model_q.save_weights("duel_dd_error600k.h5")
		print("Saved model-600k to disk")
		with open('duel_dd_error', 'wb') as er:
			pickle.dump(erro, er)

	if step == 800000:
		model_json = model_q.to_json()
		with open("duel_dd_800k.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model_q.save_weights("duel_dd_800k.h5")
		print("tarSaved model-800k to disk")

	if step == 1000000:
		model_json = model_q.to_json()
		with open("duel_dd_1000k.json", "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model_q.save_weights("duel_dd_1000k.h5")
		print("Saved model-1000k to disk")
		with open("duel_dd.json", "w") as json_file:
			json_file.write(model_json)
		# saver.save(sess, dir + '/data-all.chkp')
		with open('duel_dd_loss', 'wb') as ls:
			pickle.dump(ll, ls)
		with open('duel_dd_reward', 'wb') as rd:
			pickle.dump(re, rd)
		with open('duel_dd_term', 'wb') as tr:
			pickle.dump(term, tr)
		with open('duel_dd_step', 'wb') as ts:
			pickle.dump(st, ts)
		with open('duel_dd_error', 'wb') as er:
			pickle.dump(erro, er)					
#Used for Testing


	# if opt.disp_on:
	# 	if win_all is None:
	# 		plt.subplot(121)
	# 		win_all = plt.imshow(state.screen)
	# 		plt.subplot(122)
	# 		win_pob = plt.imshow(state.pob)
	# 	else:
	# 		win_all.set_data(state.screen)
	# 		win_pob.set_data(state.pob)
	# 	plt.pause(opt.disp_interval)
