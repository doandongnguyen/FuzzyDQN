"""
    The code is derived from the Blog: https://jaromiru.com/
"""
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Lambda
from keras.optimizers import Adam
from utils.utils import huber_loss
import globalvars
np.random.seed(globalvars.GLOBAL_SEED)


class DQN:
    def __init__(self, stateCnt, actionCnt, **kwargs):
        self.dueling = kwargs.get('dueling', False)
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.model = self.build_network()
        self.target_model = self.build_network()

    def predict(self, s, target=False):
        if not target:
            return self.model.predict(s)
        else:
            return self.target_model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt[0]),
                            target=target).flatten()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_network(self):
        inp = Input(shape=self.stateCnt)
        x = Dense(globalvars.N_HIDDENS, activation='relu')(inp)
        x = Dense(globalvars.N_HIDDENS, activation='relu')(x)
        if self.dueling:
            x = Dense(self.actionCnt + 1,
                      activation='linear')(x)
            x = Lambda(lambda i: K.expand_dims(i[:, 0],
                                               -1) + i[:, 1:] - K.mean(i[:,
                                                                       1:],
                                                            keepdims=True),
                       output_shape=(self.actionCnt,))(x)
        else:
            x = Dense(self.actionCnt, activation='linear')(x)
        model = Model(inp, x)
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=globalvars.LEARNING_RATE))
        print(model.summary())
        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=globalvars.BATCH_SIZE,
                       epochs=epochs,
                       verbose=verbose)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
