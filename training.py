import os
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import globalvars
from agents.agent import Agent
from environment import Environment
from memberships import BuildFis

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

random.seed(globalvars.GLOBAL_SEED)
np.random.seed(globalvars.GLOBAL_SEED)

DUELING = True
fis = BuildFis()
# fis = None
FILE_NAME = "DQN_" + globalvars.DEFAULT_ENV_NAME.upper()
if DUELING:
    FILE_NAME += "_DUELING"
if fis is not None:
    FILE_NAME += "_FUZZY"
FILE_NAME += '.hd5'


def training():
    gym_env = gym.make(globalvars.DEFAULT_ENV_NAME)
    gym_env = gym_env.unwrapped
    gym_env.seed(globalvars.LOCAL_SEED)
    if fis is None:
        stateCnt = gym_env.observation_space.shape
    else:
        stateCnt = fis.shape()
    actionCnt = gym_env.action_space.n
    agent = Agent(stateCnt, actionCnt,
                  dueling=DUELING)
    print('CartPole, state=', stateCnt, ' action=', actionCnt)
    env = Environment(gym_env, agent, fis=fis)
    total_eps = 150
    count = 0
    rewards = []
    while count < total_eps:
        reward = env.run()
        rewards.append(reward)
        print('Episode=', count, 'Steps=', agent.steps,
              ', reward=', reward,
              ', epsilon=', agent.epsilon)
        count += 1
    agent.save(FILE_NAME)

    plt.plot(rewards, label='Rewards')
    if fis is None:
        plt.title(globalvars.DEFAULT_ENV_NAME)
    else:
        plt.title(globalvars.DEFAULT_ENV_NAME + ' Fuzzy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    training()
