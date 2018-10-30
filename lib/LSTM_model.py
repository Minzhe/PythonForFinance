###########################################################################
###                              LSTM Model                             ###
###########################################################################
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

class LSTM_model(object):
    def __init__(self, input_shape, structure='1-layer-lstm'):
        self.input_shape = input_shape
        self.structure = structure
        self.model = self._model()
    
    def _model(self):
        print('Initilizing {} model ...'.format(self.structure), end='', flush=True)
        if self.structure == '1-layer-lstm':
            model = _1layer_lstm(input_shape=self.input_shape)
        elif self.structure == '3-layers-lstm':
            model = _3layer_lstm(input_shape=self.input_shape)
        else:
            raise ValueError('Model structure error!')
        model.compile(loss='mse', optimizer='Adam')
        print(' Done\nModel structure summary:', flush=True)
        print(model.summary())
        return model
    
    def train(self, X_train, y_train, validation_split=0.0, validation_data=None, batch_size=32, epochs=200, verbose=1, patience=None, shuffle=False):
        print('Start training neural network ... ', end='', flush=True)
        early_stopper = None if patience is None else [EarlyStopping(patience=patience, verbose=1)]
        trace = self.model.fit(X_train, y_train, 
                               validation_split=validation_split,
                               validation_data=validation_data,
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=verbose,
                               callbacks=early_stopper,
                               shuffle=shuffle)
        print('Done')
        return trace
    
    def predict_one_step(self, X):
        '''
        Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        '''
        print('Predicting 1 step ahead with LSTM model ... ', flush=True)
        y = self.model.predict(X, verbose=1)
        return y
    
    def predict_multi_steps(self, X, steps):
        '''
        Predict multi timestep given the last sequence of true data.
        '''
        print('Predicting {} step ahead with LSTM model ... '.format(steps), flush=True)
        y = self.model.predict(X, verbose=0)
        while steps - 1 > 0:
            X_new = np.concatenate([X[:,1:,:], np.reshape(y[-1], (1,-1,1))], axis=1)
            y_new = self.model.predict(X_new, verbose=0)
            y = np.concatenate([y, y_new])
            steps -= 1
        return y

    
# >>>>>>>>>>>>>>>>>>>>>>>  model structure  <<<<<<<<<<<<<<<<<<<<<<<< #
def _1layer_lstm(input_shape):
    '''
    1 Layer LSTM model.
    Stolen from https://www.kaggle.com/amarpreetsingh/stock-prediction-lstm-using-keras
    '''
    inputs = Input(shape=input_shape, name='input')
    lstm = LSTM(256) (inputs)
    output = Dense(1) (lstm)
    model = Model(inputs=inputs, outputs=output)
    return model

def _3layer_lstm(input_shape):
    '''
    3 Layer LSTM model.
    Stolen from https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/blob/master/config.json
    '''
    inputs = Input(shape=input_shape, name='input')
    lstm1 = LSTM(128, return_sequences=True) (inputs)
    drop1 = Dropout(0.2) (lstm1)
    lstm2 = LSTM(128, return_sequences=True) (drop1)
    lstm3 = LSTM(128, return_sequences=False) (lstm2)
    drop2 = Dropout(0.2) (lstm3)
    output = Dense(1) (drop2)
    model = Model(inputs=inputs, outputs = output)
    return model