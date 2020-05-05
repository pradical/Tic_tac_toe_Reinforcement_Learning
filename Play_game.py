import pandas as pd
import numpy as np
from reinforcement_learning.Tic_tac_toe.AI_Player import AI_Player
from reinforcement_learning.Tic_tac_toe.GameBoardState import GameBoardState
from reinforcement_learning.Tic_tac_toe.Human_player_interface import Human_player_interface
from reinforcement_learning.Tic_tac_toe.Play_and_train import is_victor
import re

player_state_value_path = r'C:\Users\p.b.krishna.prasad\OneDrive - Accenture\Pradyumna' \
                          r'\Python_scripts\Reinforcement_learning\reinforcement_learning' \
                          r'\Tic_tac_toe\Trial_2\PlayerA_value_state_map_13002020112019.xlsx'


def main():
    df = pd.read_excel(player_state_value_path)
    ai_player_mark = re.findall(r'(?:\bmark ([0-9]))', df.columns[1])[0]

    ai_player = AI_Player(int(ai_player_mark), learning_rate=0.1, df_value_state_map=df)
    human_player = Human_player_interface(3 - int(ai_player_mark))

    keep_playing = True
    while keep_playing:

        list_episode_df = []
        if input('Do you want to play first? Type yes or no.\n').lower() in ['yes', 'y']:
            player_1 = human_player
            player_2 = ai_player
            print('You are player 1. Your mark on the board will be {}.'.format(['X','O'][3 - int(ai_player_mark) - 1]))
        else:
            player_1 = ai_player
            player_2 = human_player
            print('You are player 2. Your mark on the board will be {}.'.format(['X','O'][3 - int(ai_player_mark) - 1]))

        current_gameboard = GameBoardState()
        df_all_episode_gameboards = pd.DataFrame(
            columns=['Step no.', 'GameBoard after player 1', 'GameBoard after player 2'])
        df_all_episode_gameboards.loc[0] = [0, 'nan', current_gameboard]

        print('The game is beginning! Initial board:\n')
        current_gameboard.draw_game_board()

        move_num = 1

        while not current_gameboard.is_board_end_state():

            # Play for the first player
            print('Player 1 move.\n')
            current_gameboard = player_1.choose_next_action_play(current_gameboard)
            current_gameboard.draw_game_board()
            df_all_episode_gameboards.loc[move_num] = [move_num, current_gameboard, 'nan']
            if current_gameboard.is_board_end_state():
                break

            # Play for the second player
            print('Player 2 move.\n')
            current_gameboard = player_2.choose_next_action_play(current_gameboard)
            current_gameboard.draw_game_board()
            df_all_episode_gameboards.iloc[move_num, 2] = current_gameboard

            move_num += 1

        if is_victor(3 - int(ai_player_mark), current_gameboard):
            print('Congratulations! You win! :) ')
        elif is_victor(int(ai_player_mark), current_gameboard):
            print('Sorry you lost. Please try again.')
        else:
            print('It\'s a draw! Please try again.')

        if input('Do you want to keep playing? Type yes or no.\n').lower() in ['no', 'n']:
            keep_playing = False



if __name__ == '__main__':
    main()
