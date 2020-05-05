import pandas as pd
import numpy as np
from reinforcement_learning.Tic_tac_toe.AI_Player import AI_Player
from reinforcement_learning.Tic_tac_toe.GameBoardState import GameBoardState
from datetime import datetime
import matplotlib.pyplot as plt

num_of_episodes = 100000
learning_rate = 0.1
win_reward = 1
draw_reward = 0.5
lose_reward = 0
default_state_value = 0.5  # Initial value of all the gameboard states
episode_save_period = 1000  # Saves the game episodes every episode_save_period

date_time_now = datetime.now().strftime("%H%M%S%d%m%Y")

save_path_value_state_map_player_a = r'C:\Users\p.b.krishna.prasad\OneDrive - Accenture\Pradyumna' \
                                     r'\Python_scripts\Reinforcement_learning\reinforcement_learning' \
                                     r'\Tic_tac_toe\Trial_2\PlayerA_value_state_map_' + date_time_now + '.xlsx'
save_path_value_state_map_player_b = r'C:\Users\p.b.krishna.prasad\OneDrive - Accenture\Pradyumna' \
                                     r'\Python_scripts\Reinforcement_learning\reinforcement_learning' \
                                     r'\Tic_tac_toe\Trial_2\PlayerB_value_state_map_' + date_time_now + '.xlsx'

save_path_episodes_gameboards = r'C:\Users\p.b.krishna.prasad\OneDrive - Accenture\Pradyumna' \
                                r'\Python_scripts\Reinforcement_learning\reinforcement_learning' \
                                r'\Tic_tac_toe\Trial_2\Episodes_gameboards_' + date_time_now + '.txt'

continue_training = True

##############
# PLAYER A MUST BE A MARK 1 PLAYER. PLAYER B MUST BE A MARK 2 PLAYER!!!
##############
player_A_state_value_path = r'C:\Users\p.b.krishna.prasad\OneDrive - Accenture\Pradyumna' \
                                     r'\Python_scripts\Reinforcement_learning\reinforcement_learning' \
                                     r'\Tic_tac_toe\Trial_2\PlayerA_value_state_map_13002020112019.xlsx'
player_B_state_value_path = r'C:\Users\p.b.krishna.prasad\OneDrive - Accenture\Pradyumna' \
                                     r'\Python_scripts\Reinforcement_learning\reinforcement_learning' \
                                     r'\Tic_tac_toe\Trial_2\PlayerB_value_state_map_10515120112019.xlsx'


def play_and_train():
    list_episode_df = []

    if continue_training:
        ##############
        # PLAYER A MUST BE A MARK 1 PLAYER. PLAYER B MUST BE A MARK 2 PLAYER!!!
        ##############
        df = pd.read_excel(player_A_state_value_path)
        df.columns = ['State index', 'Value function']
        player_a = AI_Player(player_mark=1, learning_rate=learning_rate,
                             default_state_value=default_state_value, df_value_state_map=df)

        df = pd.read_excel(player_B_state_value_path)
        df.columns = ['State index', 'Value function']
        player_b = AI_Player(player_mark=2, learning_rate=learning_rate,
                             default_state_value=default_state_value, df_value_state_map=df)

    else:
        player_a = AI_Player(player_mark=1, learning_rate=learning_rate, default_state_value=default_state_value)
        player_b = AI_Player(player_mark=2, learning_rate=learning_rate, default_state_value=default_state_value)

    player_a_rewards = np.zeros(1)
    player_b_rewards = np.zeros(1)

    # For the first loop
    player_1 = player_a
    player_2 = player_b

    # Episodes loop
    for i in range(num_of_episodes):

        if i != 0:
            player_1, player_2 = player_2, player_1

        current_gameboard = GameBoardState()
        df_all_episode_gameboards = pd.DataFrame(
            columns=['Step no.', 'GameBoard after player 1', 'GameBoard after player 2'])
        df_all_episode_gameboards.loc[0] = [0, 'nan', current_gameboard]

        move_num = 1
        is_game_draw = False

        # In game loop
        while not current_gameboard.is_board_end_state():

            # Play for the first player
            current_gameboard = player_1.choose_next_action_train(current_gameboard)
            df_all_episode_gameboards.loc[move_num] = [move_num, current_gameboard, 'nan']
            if current_gameboard.is_board_end_state():
                break

            # Play for the second player
            current_gameboard = player_2.choose_next_action_train(current_gameboard)
            df_all_episode_gameboards.iloc[move_num, 2] = current_gameboard

            move_num += 1

        if is_victor(player_a.player_mark, current_gameboard):
            player_a.update_value_functions(current_gameboard, df_all_episode_gameboards, win_reward)
            player_b.update_value_functions(current_gameboard, df_all_episode_gameboards, lose_reward)

            player_a_rewards = np.append(player_a_rewards, win_reward)
            player_b_rewards = np.append(player_b_rewards, lose_reward)

        elif is_victor(player_b.player_mark, current_gameboard):
            player_a.update_value_functions(current_gameboard, df_all_episode_gameboards, lose_reward)
            player_b.update_value_functions(current_gameboard, df_all_episode_gameboards, win_reward)

            player_a_rewards = np.append(player_a_rewards, lose_reward)
            player_b_rewards = np.append(player_b_rewards, win_reward)
        else:
            player_a.update_value_functions(current_gameboard, df_all_episode_gameboards, draw_reward)
            player_b.update_value_functions(current_gameboard, df_all_episode_gameboards, draw_reward)

            player_a_rewards = np.append(player_a_rewards, draw_reward)
            player_b_rewards = np.append(player_b_rewards, draw_reward)

        if i % episode_save_period == 0:
            list_episode_df.append([i, df_all_episode_gameboards])

        if i % 100 == 0:
            print('Completed episode {} of {}!'.format(i, num_of_episodes))

    plt.plot(moving_average(player_a_rewards, n = 200), label='Player A rewards')
    plt.plot(moving_average(player_b_rewards, n = 200), label='Player B rewards')
    plt.xlabel('Iteration number')
    plt.ylabel('Rewards')
    plt.title('Rewards vs iterations')
    plt.legend()
    plt.show()

    player_a.df_value_state_map.to_excel(save_path_value_state_map_player_a,
                                         header=['State index',
                                                 'State Values for Player A, mark {}'.format(player_a.player_mark)],
                                         index=False)
    player_b.df_value_state_map.to_excel(save_path_value_state_map_player_b,
                                         header=['State index',
                                                 'State Values for Player B, mark {}'.format(player_b.player_mark)],
                                         index=False)
    write_list_of_episodes_to_file(list_episode_df)


def is_victor(player_mark: int, current_gameboard: GameBoardState) -> bool:
    # Check if there are any marks: three in a row.
    for i in range(current_gameboard.board_matrix.shape[0]):
        if (np.all(current_gameboard.board_matrix[i, :] == current_gameboard.board_matrix[i, 0])) \
                and (current_gameboard.board_matrix[i, 0] == player_mark): return True

    # Check if there are any marks: three in a column.
    for j in range(current_gameboard.board_matrix.shape[1]):
        if np.all(current_gameboard.board_matrix[:, j] == current_gameboard.board_matrix[0, j]) \
                and (current_gameboard.board_matrix[0, j] == player_mark): return True

    # Check if there are any marks: three in any diagonal.
    if current_gameboard.board_matrix[0, 0] == current_gameboard.board_matrix[1, 1] == \
            current_gameboard.board_matrix[2, 2] == player_mark: return True
    if current_gameboard.board_matrix[0, 2] == current_gameboard.board_matrix[1, 1] == \
            current_gameboard.board_matrix[2, 0] == player_mark: return True

    return False


def write_list_of_episodes_to_file(list_episode_df: list):
    write_array = np.full((2, 3), 'nan')

    for episode in list_episode_df:
        write_array = np.append(write_array, [np.array(['nan', 'nan', 'Episode num {}'.format(episode[0])])], axis=0)

        for i_row in range(episode[1].shape[0]):
            if episode[1].iloc[i_row, 1] != 'nan':
                write_array = np.append(write_array, episode[1].iloc[i_row, 1].board_matrix, axis=0)
            write_array = np.append(write_array, np.full((1, 3), 'nan'), axis=0)

            if episode[1].iloc[i_row, 2] != 'nan':
                write_array = np.append(write_array, episode[1].iloc[i_row, 2].board_matrix, axis=0)
            write_array = np.append(write_array, np.full((1, 3), 'nan'), axis=0)

    np.savetxt(save_path_episodes_gameboards, write_array, fmt='%s')


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    play_and_train()
