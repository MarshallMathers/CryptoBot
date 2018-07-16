import numpy as np
import keras
from keras.layers import Input, Dense, Flatten, Activation
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.agents import DQNAgent
from qEnv import trader

episode_length = 2000
trading_fee = 0.2
time_fee = 0.0
# history_length number of historical states in the observation vector.
history_length = 2
genData = '../largeData/data.data'

env = trader(fn=genData,
                            trading_fee=trading_fee,
                            time_fee=time_fee,
                            history_length=history_length,
                            episode_length=episode_length)

nb_actions = env.n_actions
model = keras.Sequential()
model.add(Flatten(input_shape=(1,) + env.state_shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=200000, window_length=1)
policy = EpsGreedyQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=200000, visualize=False, verbose=2)
print(env._total_pnl)
# After training is done, we save the final weights.
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
dqn.save_weights('dqn_{}_weights.h5f'.format('trader'), overwrite=True)


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)