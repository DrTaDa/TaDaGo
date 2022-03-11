
import numpy as np
import copy

import valueNet
import policyNet
import utils


class TaDaGo():

    """TaDaGo Agent"""

    def __init__(self, color, policy_weights_path, value_weights_path):
        
        if color in ['b', 'w']:
            self.color = color
        else:
            raise ValueError
            
        self.policy_net = policyNet.PolicyNet(policy_weights_path)
        self.value_net = valueNet.ValueNet(value_weights_path)
        
        self.old_boards = [np.full((19,19), None), np.full((19,19), None), np.full((19,19), None)]
    
    def save_board(self, board):
    
        self.old_boards = [np.array(self.old_boards[-2]), 
                           np.array(self.old_boards[-1]), 
                           np.array(board.board)]
    
    def play_move(self, board):
        
        # Send the board to the policy_net and get the most likely next moves
        proposed_moves = self.policy_net.predict_from_board(self.old_boards, self.color)
        
        # Set the probability to zero for the moves that are impossible
        for idx in range(361):
            play = [int(idx/19), idx%19]
            if self.old_boards[-1][play[0], play[1]] != None:
                 proposed_moves[idx] = 0.

        # Send the 10 most likely moves to the value network and get its opinion
        sorted_moves_idx = np.argsort(proposed_moves)[::-1]
        values = np.zeros(362)
        for i, idx in enumerate(sorted_moves_idx):
        
            if i >= 50:
                break

            # Play the move and evaluate the win likelihood for the color of the agent
            play = [int(idx/19), idx%19]
            play = (self.color, play)
            
            temp_board = copy.deepcopy(board)
            temp_board = utils.play_turn_train(temp_board, play)
            
            pred = self.value_net.predict_from_board(temp_board, next_player=self.color)
            if self.color == 'b':
                values[idx] = pred[0]
            else:
                values[idx] = pred[1]

        # The final score for each move is how likely it is to be played times how good the new board is
        scores = proposed_moves * values
        
        play = np.argmax(scores)
        play = [int(play/19), play%19]
        
        return [self.color, play], proposed_moves, values, scores