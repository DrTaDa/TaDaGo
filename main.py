
import glob

import numpy as np
import tensorflow as tf

from sgfmill import ascii_boards
from sgfmill import sgf
from sgfmill import sgf_moves

import valueNet
import policyNet

import tadago
import utils


sgf_paths = glob.glob("jgdb/sgf/**/*.sgf", recursive=True)

value_net = valueNet.ValueNet()
value_net.train(sgf_paths, batch_size=512, epochs=200, lr=0.1, freeze=None)
value_net = valueNet.ValueNet("weights_ValueNet.h5")
value_net.train(sgf_paths, batch_size=512, epochs=200, lr=0.01, freeze=None)
value_net = valueNet.ValueNet("weights_ValueNet.h5")
value_net.train(sgf_paths, batch_size=512, epochs=200, lr=0.001, freeze=None)
value_net = valueNet.ValueNet("weights_ValueNet.h5")
value_net.train(sgf_paths, batch_size=512, epochs=100, lr=0.0001, freeze=None)

# print(len(value_net.model.layers))
# for i in range(70):
    # value_net.model.layers[i].set_weights(value_net_sig.model.layers[i].get_weights())

# Play
# human_color = 'w'
# bot_color = 'b'
# bot = tadago.TaDaGo(color=bot_color, 
             # policy_weights_path='weights_PolicyNet_3.30906.h5', 
             # value_weights_path='weights_ValueNet.h5')


# file_path = "jgdb/sgf/val/0002/00002000.sgf"
# with open(file_path, "rb") as fp:
    # sgf_src = fp.read()
# sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
# sgf_board, plays = sgf_moves.get_setup_and_moves(sgf_game)


# turn = 0
# while True:
        
    # if turn%2 == 0:
        # next_player = 'b'
    # else:
        # next_player = 'w'
        
    # if next_player == bot_color:
        # play, proposed_moves, values, scores = bot.play_move(sgf_board)
        # print("Bot plays: ", play)
        # bla = utils.draw_debug_board(sgf_board, proposed_moves, values, scores)
        # sgf_board = utils.play_turn_play(sgf_board, play)

    # else:
        # play = utils.draw_board_play(sgf_board)
        # play = [human_color, play]
        # print("Other player plays: ", play)
        # sgf_board = utils.play_turn_play(sgf_board, play)
    
    # bot.save_board(sgf_board)
    
    # turn += 1
    
    