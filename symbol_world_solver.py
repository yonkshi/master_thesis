'''
This program solves the symbol world problem from the symbol world environment
'''

import sys
import time
sys.path.append('/Users/Yonk/school/symbolic_gridworld')

from PyQt5.QtCore import Qt
import numpy as np
import h5py
import gym
import pycolab
import matplotlib.pyplot as plt

def main():

    env = gym.make('SymbolWorld-v0')

    obs = env.reset()
    reward = 0
    print('hello world')



    with h5py.File('data/SymWorld_basic.hdf5', "w") as f:
        ep = 0
        # Loop through episodes
        while ep < 10000:
            if ep % 10 == 0 and ep > 0: print(ep)

            rewards_set = []
            obs_set = []
            action_set = []

            # Loop through each step
            while True:

                action = solve(obs, env)

                action_vec = action2vec(action, env)
                action_set.append(action_vec)
                obs_set.append(obs.board)
                rewards_set.append(reward)


                obs, reward, done, info = env.step(action)

                if not reward: reward = 0

                # End of an episode
                if done:
                    break

            # add final observation
            obs_set.append(obs.board)

            f.create_dataset(f"symworld/ep{ep}/obs_set", data=obs_set, dtype='uint8')
            f.create_dataset(f"symworld/ep{ep}/reward_set", data=rewards_set, dtype='int8')
            f.create_dataset(f"symworld/ep{ep}/action_set", data=action_set, dtype='uint8')

            rewards_set = []
            obs_set = []
            action_set = []
            obs = env.reset()
            ep += 1




def solve(obs, env):
    sym_board = obs.symbolic_board

    PLAYER = 80
    KEY = 75
    ADV = 97
    GOAL = 64

    player_pos = np.argwhere(sym_board == PLAYER)[0]
    keys_pos = np.argwhere(sym_board == KEY)
    goals_pos = np.argwhere(sym_board == GOAL)
    if len(np.argwhere(sym_board == ADV)) == 0:
        adv_pos = [0, 0]
    else:
        adv_pos = np.argwhere(sym_board == ADV)[0]

    def dir(coord):
        if coord[0] > 0:
            return env.actions.down
        elif coord[0] < 0:
            return env.actions.up
        elif coord[1] > 0:
            return env.actions.right
        elif coord[1] < 0:
            return env.actions.left


    # Avoid adversary logic, assuming only single adversary
    adv_dist = np.linalg.norm(adv_pos - player_pos)
    if adv_dist <= 10:
        diff = adv_pos - player_pos
        return dir(-diff)

    key_dist = np.linalg.norm(keys_pos - player_pos, axis = 1)
    goal_dist = np.linalg.norm(goals_pos - player_pos, axis=1)

    # Check if we need to retrieve keys first OR if the closest object is a key
    if(len(key_dist) >= len(goal_dist)) or (len(key_dist) > 0 and np.min(key_dist) < np.min(goal_dist)):
        closest_key = keys_pos[np.argmin(key_dist)]
        return dir(closest_key - player_pos)
    else:
        closest_goal = goals_pos[np.argmin(goal_dist)]
        return dir(closest_goal - player_pos)

def action2vec(action, env):
    '''Converts a gym Env action into a vector'''
    vec = np.zeros(env.action_space.n)
    vec[action] = 1
    return vec




if __name__ == '__main__':
    main()