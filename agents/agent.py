"""
    The code is derived from the Blog: https://jaromiru.com/
"""
import numpy as np
import random
import math
from agents.memory import Memory
from agents.dqn import DQN
import globalvars

np.random.seed(globalvars.GLOBAL_SEED)


class Agent:
    def __init__(self, stateCnt, actionCnt, **kwargs):
        if 'state_1d' in kwargs:
            state_1d = kwargs['state_1d']
        else:
            state_1d = False
        if 'dueling' in kwargs:
            dueling = kwargs['dueling']
        else:
            dueling = False
        self.steps = 0
        self.epsilon = globalvars.MAX_EPSILON
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.dqn = DQN(self.stateCnt, self.actionCnt,
                       state_1d=state_1d,
                       dueling=dueling)
        self.memory = Memory()

    def acts(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return np.argmax(self.dqn.predictOne(s))

    def observe(self, sample):
        if self.steps <= globalvars.REPLAY_START_SIZE:
            error = abs(sample[2])
            self.memory.add(error, sample)
        else:
            x, y, a, errors = self._getTargets([(0, sample)])
            self.memory.add(errors[0], sample)
            if self.steps % globalvars.SYNC_TARGET == 0:
                self.dqn.update_target_model()
            # Epsilon decay
            self.epsilon = globalvars.MIN_EPSILON + \
                           (globalvars.MAX_EPSILON - globalvars.MIN_EPSILON) * math.exp(-globalvars.LAMBDA * (self.steps \
                                                                                                              - globalvars.REPLAY_START_SIZE))
        self.steps += 1

    def _getTargets(self, batch):
        states = np.array([o[1][0] for o in batch])
        if len(self.stateCnt) > 1:
            no_state = np.zeros(self.stateCnt)
        else:
            no_state = np.zeros(self.stateCnt[0])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) \
                            for o in batch])
        p = self.dqn.predict(states)
        p_ = self.dqn.predict(states_, target=False)
        pTarget_ = self.dqn.predict(states_, target=True)

        if len(self.stateCnt) > 1:
            x = np.zeros(states.shape)
        else:
            x = np.zeros((len(batch), self.stateCnt[0]))
        y = np.zeros((len(batch), self.actionCnt))
        errors = np.zeros(len(batch))
        actions = []
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + globalvars.GAMMA * pTarget_[i][np.argmax(p_[i])]
            x[i] = s
            y[i] = t
            actions.append(a)
            errors[i] = abs(oldVal - t[a])
        return x, y, a, errors

    def replay(self):
        batch = self.memory.sample(globalvars.BATCH_SIZE)
        x, y, a, errors = self._getTargets(batch)
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])
        self.dqn.train(x, y)

    def save(self, name):
        self.dqn.save(name)
        print('Saved model to ', name)

    def load(self, name):
        self.dqn.load(name)
        print('Loaded model from ', name)
