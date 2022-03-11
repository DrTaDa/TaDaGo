
import numpy as np

from keras.layers import Input, Flatten, Conv2D, Dense, ReLU, add, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle

from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves

import utils


class ValueNet():

    """Value network"""
    
    def __init__(self, weights_path=False):
    
        inputs = Input(shape=(3, 19, 19, ))
         
        # Convolutional block
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Residual block
        for i in range(11):
            x = self.residual_block(x)
        
        # Value head
        v = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        v = BatchNormalization()(v)
        v = ReLU()(v)
        v = Flatten()(v)
        v = Dense(256, kernel_regularizer=regularizers.l2(0.0001))(v)
        v = ReLU()(v)
        v = Dense(1, activation = 'tanh', kernel_regularizer=regularizers.l2(0.0001))(v)

        self.model = Model(inputs=inputs, outputs=v)
        
        if weights_path:
            self.model.load_weights(weights_path)

    def residual_block(self, s):
        
        # Residual block
        shortcut = s
        s = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001))(s)
        s = BatchNormalization()(s)
        s = ReLU()(s)
        s = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001))(s)
        s = BatchNormalization()(s)
        s = add([shortcut, s])
        s = ReLU()(s)
            
        return s
    
    def loss(self, y_true, y_pred):

        return K.categorical_crossentropy(y_true, y_pred)

    def predict_from_board(self, board, next_player):
        
        if next_player not in ['b', 'w']:
            raise ValueError
        
        np_board = np.asarray(board.board)
        b_board = (np_board == "b").astype(int)
        w_board = (np_board == "w").astype(int)
        
        if next_player == 'w':
            whose_turn = np.zeros((19,19))
        else:
            whose_turn = np.full((19,19), 1)

        # Create the feature array black board, white board, next player
        features = np.stack([b_board, w_board, whose_turn], axis=0)
        features = np.expand_dims(features, axis=0)
        
        pred = self.model.predict(features)

        return pred[0]

    def generator(self, batch_size, paths):
    
        current_idx = 0
        while True:

            data_to_return, label_to_return = [], []
            while len(data_to_return) < batch_size:
                path = paths[current_idx]
                data, label = self.read_sgf(path)
                if data is not False:
                    data_to_return.append(data)
                    label_to_return.append(label)
                current_idx += 1
                if current_idx >= len(paths):
                    current_idx = 0
                    
            yield np.asarray(data_to_return), np.asarray(label_to_return)
    
    def read_sgf(self, file_path):
          
        # Open the game and setup the sgf object
        with open(file_path, "rb") as fp:
            sgf_src = fp.read()
            
        try:
            sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
            board, plays = sgf_moves.get_setup_and_moves(sgf_game)
        except ValueError:
            return False, False

        # Choose a turn at random from which the net will have to predict the winner
        length_game = len(plays)
        if length_game:
            turn = np.random.randint(0, length_game)
        else:
            return False, False

        # Read the game until the play
        _ = plays[:turn]
        for play in _:
            board = utils.play_turn_train(board, play)
            if board is False:
                return False, False

        np_board = np.asarray(board.board)
        b_board = (np_board == "b").astype(int)
        w_board = (np_board == "w").astype(int)
        
        if plays[turn][0] == 'b':
            whose_turn = np.full((19,19), 1)
        elif plays[turn][0] == 'w':
            whose_turn = np.zeros((19,19))
        else:
            return False, False

        # Create the feature array black board, white board, next player
        features = np.stack([b_board, w_board, whose_turn], axis=0)
 
        # Label is one-hot of winner: black/white
        winner = utils.get_winner(file_path)
        if winner == "b":
            label = [1.]
        elif winner == "w":
           label = [-1.]
        else:
            return False, False

        return features, label
    
    def train(self, sgf_paths, batch_size=2048, epochs=100, lr=0.01, freeze=None):
    
        gen_train = self.generator(batch_size, shuffle(sgf_paths))
        gen_test = self.generator(batch_size, shuffle(sgf_paths))
        
        optimizer = SGD(lr=lr, momentum=0.9, decay=0., nesterov=True, clipnorm=1.)
        checkp = ModelCheckpoint(filepath="weights_ValueNet.h5", verbose=1, save_best_only=True)
        
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'acc'])
        hist = self.model.fit_generator(gen_train,
                                        steps_per_epoch = 1000,
                                        epochs = epochs,
                                        shuffle = True,
                                        verbose = 1,
                                        validation_data = gen_test,
                                        validation_steps = 50,
                                        callbacks = [checkp])
                                        
        return hist

    def test_accuracy(self, sgf_paths, sample_size=1000):
    
        correct = 0
        total = 0
        
        while total != sample_size:

            path = np.random.choice(sgf_paths, 1)[0]

            try:

                with open(path, "rb") as fp:
                    sgf_src = fp.read()
                sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
                board, plays = sgf_moves.get_setup_and_moves(sgf_game)

                length_game = len(plays)
                turn = np.random.randint(0, length_game)
                _ = plays[:turn]
                for play in _:
                    board = utils.play_turn_train(board, play)

                pred = self.predict_from_board(board, plays[turn][0])
                print(pred)
                pred_winner = np.argmax(pred)
                
                winner = utils.get_winner(path)
                label = 0 if winner == "b" else 1
                
                if pred_winner == label:
                    correct += 1
                total += 1

            except:
                pass
        
        return correct/total
    