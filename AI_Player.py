import pandas as pd
from reinforcement_learning.Tic_tac_toe.GameBoardState import GameBoardState
import numpy as np


class AI_Player:

    def __init__(self,
                 player_mark: int,
                 learning_rate,
                 default_state_value=0.5,
                 df_value_state_map=pd.DataFrame({'State index': np.array(range(19683)),
                                                  'Value function': np.full(19683, 0.5)}),
                 episodes_played=0,
                 ):

        self.player_mark: int = player_mark
        assert player_mark in [1, 2]
        self.default_state_value = default_state_value
        self.learning_rate = learning_rate
        self._episodes_played = episodes_played

        self.df_value_state_map = df_value_state_map

    def reset_value_functions(self):
        self.df_value_state_map['Value function'] = np.full(19683, 0.5)

    def update_value_functions(self, end_state_board: GameBoardState, df_all_episode_gameboards: pd.DataFrame, reward):
        self._episodes_played += 1
        num_rows = df_all_episode_gameboards.shape[0]

        # Update the state values of the last row
        if df_all_episode_gameboards.iloc[num_rows - 1, 2] != 'nan':
            current_state = df_all_episode_gameboards.iloc[num_rows - 1, 2].get_board_state_index()
            self.df_value_state_map.iloc[current_state, 1] = reward
            next_state = current_state

            current_state = df_all_episode_gameboards.iloc[num_rows - 1, 1].get_board_state_index()
            self.df_value_state_map.iloc[current_state, 1] += \
                self.learning_rate * (
                            self.df_value_state_map.iloc[next_state, 1] - self.df_value_state_map.iloc[current_state, 1])
        else:
            current_state = df_all_episode_gameboards.iloc[num_rows - 1, 1].get_board_state_index()
            self.df_value_state_map.iloc[current_state, 1] = reward

        for idx_row in range(2, num_rows):
            # Update state value for 'Gameboard after player 2
            current_state = df_all_episode_gameboards.iloc[num_rows - idx_row, 2].get_board_state_index()
            next_state = df_all_episode_gameboards.iloc[num_rows - idx_row + 1, 1].get_board_state_index()
            self.df_value_state_map.iloc[current_state, 1] += \
                self.learning_rate * (
                        self.df_value_state_map.iloc[next_state, 1] - self.df_value_state_map.iloc[current_state, 1])

            # Update state value for 'Gameboard after player 1
            current_state = df_all_episode_gameboards.iloc[num_rows - idx_row, 1].get_board_state_index()
            next_state = df_all_episode_gameboards.iloc[num_rows - idx_row, 2].get_board_state_index()
            self.df_value_state_map.iloc[current_state, 1] += \
                self.learning_rate * (
                        self.df_value_state_map.iloc[next_state, 1] - self.df_value_state_map.iloc[current_state, 1])

    def choose_next_action_train(self, current_gameboard: GameBoardState) -> GameBoardState:
        rand = np.random.random()
        list_of_children_boards = current_gameboard.get_children_states_for_player(self.player_mark)

        if rand < 1 / ((self._episodes_played + 1) ** 0.25):
            return list_of_children_boards[np.random.choice(range(0, len(list_of_children_boards)))]
        else:
            j = np.argmax([self.get_value_of_gameboard_state(child) for child in list_of_children_boards])
            return list_of_children_boards[j]

    def choose_next_action_play(self, current_gameboard: GameBoardState) -> GameBoardState:
        list_of_children_boards = current_gameboard.get_children_states_for_player(self.player_mark)
        j = np.argmax([self.get_value_of_gameboard_state(child) for child in list_of_children_boards])
        return list_of_children_boards[j]

    def get_value_of_gameboard_state(self, in_gameboard: GameBoardState) -> int:
        return self.df_value_state_map.iloc[in_gameboard.get_board_state_index(), 1]

    def save_state_value_map(self):
        raise NotImplementedError

    def create_gameboard_from_state_index(self, state_number) -> GameBoardState:
        raise NotImplementedError
