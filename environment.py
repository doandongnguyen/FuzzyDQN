import numpy as np
import globalvars

np.random.seed(globalvars.GLOBAL_SEED)


class Environment:
    def __init__(self, env, agent, fis=None):
        self.env = env
        self.agent = agent
        self.fis = fis

    def run(self):
        s = self.env.reset()
        if self.fis is not None:
            s = self.fis.get_truth_values(s)
        R = 0
        while True:
            a = self.agent.acts(s)
            s_, r, done, info = self.env.step(a)
            if self.fis is not None:
                s_ = self.fis.get_truth_values(s_)
            # Change rewards
            if r == 0:
                r = -1
            if done:
                s_ = None
            self.agent.observe((s, a, r, s_))
            self.agent.replay()
            s = s_
            R += r
            if abs(R) > 2000:
                done = True
            if done:
                break
        return R
