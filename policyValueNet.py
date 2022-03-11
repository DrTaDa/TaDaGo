
import numpy as np
import tensorflow as tf

from keras.layers import Input, Flatten, Conv2D, Dense, ReLU, add
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


class PolicyValueNet():

    """Policy network"""
    
    def __init__(self, weights_path=False):
        
        inputs = Input(shape=(7, 19, 19, ))
         
        # Convolutional block
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Residual block
        for i in range(13):
            x = self.residual_block(x)
        
        # Policy head
        p = Conv2D(2, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        p = BatchNormalization()(p)
        p = ReLU()(p)
        p = Flatten()(p)
        p = Dense(362, activation = 'softmax', kernel_regularizer=regularizers.l2(0.0001), name="policy")(p)
        
        # Value head
        v = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        v = BatchNormalization()(v)
        v = ReLU()(v)
        v = Flatten()(v)
        v = Dense(256, kernel_regularizer=regularizers.l2(0.0001))(v)
        v = ReLU()(v)
        v = Dense(2, activation = 'softmax', kernel_regularizer=regularizers.l2(0.0001))(v)
        
        self.model = Model(inputs=inputs, outputs=[p, v])
        
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

    def policy_loss(self, y_true, y_pred):

        return K.categorical_crossentropy(y_true, y_pred)
        
    def predict_from_board(self, boards, agent_color):
        
        if agent_color not in ['b', 'w']:
            raise ValueError
        
        features = np.zeros((7, 19,19))
        
        if agent_color == 'b':
            other_color = 'w'
            features[6] = np.full((19,19), 1)
        else:
            other_color = 'b'
            features[6] = np.zeros((19,19))

        features[0] = (boards[-1] == agent_color).astype(int)
        features[1] = (boards[-2] == agent_color).astype(int)
        features[2] = (boards[-3] == agent_color).astype(int)
        features[3] = (boards[-1] == other_color).astype(int)
        features[4] = (boards[-2] == other_color).astype(int)
        features[5] = (boards[-3] == other_color).astype(int)

        features = np.reshape(features, (1,7,19,19))
        
        return self.model.predict(features)[0]

    def read_sgf(self, file_path):

        # Open the game and setup the sgf object
        with open(file_path, "rb") as fp:
            sgf_src = fp.read()
        try:
            sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
            board, plays = sgf_moves.get_setup_and_moves(sgf_game)
        except ValueError:
            return False, False, False
            
        # Choose a turn at random (that turn will be the label, the turn to predict)
        length_game = len(plays)
        if length_game:
            turn = np.random.randint(0, length_game)
        else:
            return False, False, False
            
        if not(turn%2):
            whose_turn = np.full((19,19), 1)
            current_player = "b"
            other_player = "w"
        else: 
            whose_turn = np.zeros((19,19))
            current_player =  "w"
            other_player = "b"
        
        # cp = current player, op = other player
        
        # Read the game until 2 turns before the play
        if turn-3 < 0:
            cp_m2 = (np.zeros((19,19))).astype(int)
            op_m2 = (np.zeros((19,19))).astype(int)
        else:
            _ = plays[:turn-2]
            for play in _:
                board = utils.play_turn_train(board, play)
                if board is False:
                    return False, False, False
            np_board = np.asarray(board.board)
            cp_m2 = (np_board == current_player).astype(int)
            op_m2 = (np_board == other_player).astype(int)

        if turn-2 < 0:
            cp_m1 = (np.zeros((19,19))).astype(int)
            op_m1 = (np.zeros((19,19))).astype(int)
        else:
            board = utils.play_turn_train(board, plays[turn-2])
            if board is False:
                return False, False, False
            np_board = np.asarray(board.board)
            cp_m1 = (np_board == current_player).astype(int)
            op_m1 = (np_board == other_player).astype(int)
        
        if turn-1 < 0:
            cp = (np.zeros((19,19))).astype(int)
            op = (np.zeros((19,19))).astype(int)
        else:
            board = utils.play_turn_train(board, plays[turn-1])
            if board is False:
                return False, False, False
            np_board = np.asarray(board.board)
            cp = (np_board == current_player).astype(int)
            op = (np_board == other_player).astype(int)
        
        features = np.stack([cp, op, cp_m1, op_m1, cp_m2, op_m2, whose_turn], axis=0)
        
        label_policy = np.zeros(362)
        colour = plays[turn][0]  
        move = plays[turn][1]
        if move is None:
            return False, False, False
        else:
            row, col = move
            try:
                board.play(row, col, colour)
            except ValueError:
                return False, False, False
            if colour != current_player:
                return False, False, False
            label_policy[row*19 + col] = 1

        # Label is one-hot of winner: black/white
        winner = utils.get_winner(file_path)
        if winner == current_player:
            label_value = 1.
        elif winner == other_player:
           label_value = -1.
        else:
            return False, False, False

        return features, label_policy, label_value
    
    def generator(self, batch_size, paths):
    
        current_idx = 0
        while True:

            data_to_return, label_p, label_v = [], [], []
            while len(data_to_return) < batch_size:

                path = paths[current_idx]
                data, lp, lv = self.read_sgf(path)
                
                if data is not False:
                    data_to_return.append(data)
                    label_p.append(lp)
                    label_v.append(lv)
                    
                current_idx += 1
                if current_idx >= len(paths):
                    current_idx = 0
            
            yield (np.asarray(data_to_return), {'policy': np.asarray(label_p), 'value': np.asarray(label_v)})

    def train(self, sgf_paths, batch_size=64, epochs=100, lr=0.01, freeze=None):
    
        gen_train = self.generator(batch_size, shuffle(sgf_paths))
        gen_test = self.generator(batch_size, shuffle(sgf_paths))
        
        optimizer = SGD(lr=lr, momentum=0.9, decay=0., nesterov=True, clipnorm=1.)
        checkp = ModelCheckpoint(filepath="weights.h5", verbose=1, save_best_only=True)
        
        loss_weights = {'policy': 0.5, 'value': 0.5}
        loss = {'policy': self.policy_loss, 'value': 'mean_squared_error'}
        self.model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)   
        hist = self.model.fit_generator(gen_train,
                                        steps_per_epoch = 500,
                                        epochs = epochs,
                                        shuffle = False,
                                        verbose = 1,
                                        validation_data = gen_test,
                                        validation_steps = 10,
                                        callbacks = [checkp],
                                        workers=1,
                                        use_multiprocessing=False)

        return hist
