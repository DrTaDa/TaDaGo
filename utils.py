
import codecs

import numpy as np
import cv2


def get_winner(file_path):
    
    with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as f:
        for l in f.readlines():
            idx = l.find('RE')
            if idx != -1:
                winner = l[idx+3].lower()
                if winner == 'b' or winner == 'w':
                    return winner
                else:
                    return None
        else:
            return None
            
            
def play_turn_train(board, play):

    colour = play[0]
    move = play[1]
    
    if move is None:
        return False
        
    try:
        row, col = move
        board.play(row, col, colour)
    except:
        return False
        
    return board


refPt = []
def click_and_crop(event, x, y, flags, param):

    global refPt
    if event == cv2.EVENT_LBUTTONUP:
        refPt = [x, y]
        
        
def play_turn_play(board, play):

    colour = play[0]
    move = play[1]
    if move is None:
        pass
        return board
    row, col = move
    board.play(row, col, colour)
    return board

    
def draw_board(x, label, winner=False, pred_policy=False, pred_value=False):
    
    print("###################################################")
    
    if np.max(x[-1]) == 1:
        current_player = 'b'
        other_player = 'w'
        print("It is Black's turn to play")
    else:
        current_player = 'w'
        other_player = 'b'
        print("It is White's turn to play")
    
    if winner is not False:
        if winner == current_player:
            print("The winner of the game is {}".format(current_player))
        else:
            print("The winner of the game is {}".format(other_player))
            
    if pred_value is not False:
        pred_value = (pred_value+1.)/2.
        if pred_value > 0.5:
            print("The value network estimates that {} has a {}% chance of winning.".format(other_player, pred_value*100.))
        else:
            print("The value network estimates that {} has a {}% chance of winning.".format(current_player, pred_value*100.))

    scale = 37
    offset = int(scale/2.)
    
    board = np.full((scale*19,scale*19, 3), (0,153,204))
    for i in range(19):
        cv2.line(board, ((scale*i)+offset, offset), ((scale*i)+offset, (scale*18)+offset), (0,100,150), 3)
    for i in range(19):
        cv2.line(board, (offset, (scale*i)+offset), ((scale*18)+offset, (scale*i)+offset), (0,100,150), 3)
    stars = [[3,3], [15, 15], [9, 9], [3, 15], [15, 3], [15, 9], [9, 3], [9, 15], [3, 9]]
    for s in stars:
        cv2.circle(board, (int(s[0]*scale)+offset, int(s[1]*scale)+offset), 7, (0,90,140), -1)

    if current_player == 'b':
        stone_b = np.transpose(np.nonzero(x[0]))
        stone_w = np.transpose(np.nonzero(x[3]))
    else:
        stone_b = np.transpose(np.nonzero(x[3]))
        stone_w = np.transpose(np.nonzero(x[0]))  
        
    for p in stone_b:
        cv2.circle(board, (int(p[0]*scale)+offset, int(p[1]*scale)+offset), offset, (255,255,255), -1)
    for p in stone_w:
        cv2.circle(board, (int(p[0]*scale)+offset, int(p[1]*scale)+offset), offset, (0,0,0), -1)
    
    label = np.reshape(label[:-1], (19,19))
    _ = np.unravel_index(np.argmax(label), label.shape) 
    cv2.circle(board, (int(_[0]*scale)+offset, int(_[1]*scale)+offset), offset-5, (0,255,0), 6)          

    if pred_policy is not False:
        pred_policy = np.reshape(pred_policy[:-1], (19,19))
        print("The policy network predicts the plays: ")
        depth_max = 9
        for i in range(depth_max):
            _ = np.unravel_index(np.argmax(pred_policy), pred_policy.shape)
            if np.max(pred_policy) > 0.001:
                print(i+1, np.max(pred_policy))
                cv2.putText(board, '{}'.format(i+1), (int(_[0]*scale)+10, int(_[1]*scale)+26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,170,255), 3, cv2.LINE_AA)
            pred_policy[_[0], _[1]] = 0.

    cv2.imshow('', board.astype("uint8"))
    cv2.waitKey(0)

    return 0


def draw_debug_board(sgf_board, proposed_moves, values, scores):
    
    global refPt
    scale = 50
    offset = int(scale/2.)
    
    canvas = np.full((scale*19,scale*19, 3), (0,153,204))
    for i in range(19):
        cv2.line(canvas, ((scale*i)+offset, offset), ((scale*i)+offset, (scale*18)+offset), (0,100,150), 3)
        cv2.line(canvas, (offset, (scale*i)+offset), ((scale*18)+offset, (scale*i)+offset), (0,100,150), 3)
    stars = [[3,3], [15, 15], [9, 9], [3, 15], [15, 3], [15, 9], [9, 3], [9, 15], [3, 9]]
    for s in stars:
        cv2.circle(canvas, (int(s[0]*scale)+offset, int(s[1]*scale)+offset), 7, (0,90,140), -1)
    
    for i in range(19):
        cv2.putText(canvas, '{}'.format(i+1), (int(0.06*offset), (scale*i)+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,150), 1, cv2.LINE_AA)
        cv2.putText(canvas, '{}'.format(i+1), ((scale*i)+offset, int(0.54*offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,150), 1, cv2.LINE_AA)
    
    for i in range(19):
        for j in range(19):
            if sgf_board.board[i][j] == 'b':
                cv2.circle(canvas, (int(j*scale)+offset, int(i*scale)+offset), offset, (0,0,0), -1)
            elif sgf_board.board[i][j] == 'w':
                cv2.circle(canvas, (int(j*scale)+offset, int(i*scale)+offset), offset, (255,255,255), -1)
    
    # Plot Policy and values
    idxs = np.argsort(proposed_moves)[::-1]
    depth_max = 50
    for i in range(depth_max):
        idx = idxs[i]
        proba = proposed_moves[idx]
        pos = [idx%19, int(idx/19)]
        cv2.putText(canvas, '{:.0e}'.format(proba), (int(pos[0]*scale)+10, int(pos[1]*scale)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,0,255), 1, cv2.LINE_AA)
        
        value = values[idx]
        cv2.putText(canvas, '{:.2f}'.format(value), (int(pos[0]*scale)+10, int(pos[1]*scale)+35), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,255,0), 1, cv2.LINE_AA)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", canvas.astype("uint8"))
        cv2.waitKey(1)
        if len(refPt) == 2:
            x = int(round((refPt[0]-offset)/float(scale)))
            y = int(round((refPt[1]-offset)/float(scale)))
            play = [y,x]
            break

    cv2.destroyAllWindows()
    refPt = []
    return play

    
def draw_board_play(sgf_board):
    
    global refPt
    scale = 50
    offset = int(scale/2.)
    
    canvas = np.full((scale*19,scale*19, 3), (0,153,204))
    for i in range(19):
        cv2.line(canvas, ((scale*i)+offset, offset), ((scale*i)+offset, (scale*18)+offset), (0,100,150), 3)
        cv2.line(canvas, (offset, (scale*i)+offset), ((scale*18)+offset, (scale*i)+offset), (0,100,150), 3)
    stars = [[3,3], [15, 15], [9, 9], [3, 15], [15, 3], [15, 9], [9, 3], [9, 15], [3, 9]]
    for s in stars:
        cv2.circle(canvas, (int(s[0]*scale)+offset, int(s[1]*scale)+offset), 7, (0,90,140), -1)
    
    for i in range(19):
        cv2.putText(canvas, '{}'.format(i+1), (int(0.06*offset), (scale*i)+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,150), 1, cv2.LINE_AA)
        cv2.putText(canvas, '{}'.format(i+1), ((scale*i)+offset, int(0.54*offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,100,150), 1, cv2.LINE_AA)
    
    for i in range(19):
        for j in range(19):
            if sgf_board.board[i][j] == 'b':
                cv2.circle(canvas, (int(j*scale)+offset, int(i*scale)+offset), offset, (0,0,0), -1)
            elif sgf_board.board[i][j] == 'w':
                cv2.circle(canvas, (int(j*scale)+offset, int(i*scale)+offset), offset, (255,255,255), -1)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    while True:
        cv2.imshow("image", canvas.astype("uint8"))
        cv2.waitKey(1)
        if len(refPt) == 2:
            x = int(round((refPt[0]-offset)/float(scale)))
            y = int(round((refPt[1]-offset)/float(scale)))
            play = [y,x]
            break

    cv2.destroyAllWindows()
    refPt = []
    return play

