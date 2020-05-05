df = pd.DataFrame({'State index': np.array(range(3)),
                                                  'Value function': np.full(3, 0.5)})


df.shape[0]

df.iloc[2, 0]


df_all_episode_gameboards.loc[2, 0] = nan


def update_value_functions(self, end_state_board: GameBoardState, df_all_episode_gameboards: pd.DataFrame, reward):
    self._episodes_played += 1
    num_rows = df_all_episode_gameboards.shape[0]

    if df_all_episode_gameboards.iloc[num_rows - 1, 2] != 'nan':
        current_state = df_all_episode_gameboards.iloc[num_rows - 1, 2].get_board_state_index()
        self.df_value_state_map.iloc[current_state, 1] = reward
        next_state = current_state

        current_state = df_all_episode_gameboards.iloc[num_rows - 1, 1].get_board_state_index()
        self.df_value_state_map.iloc[current_state, 1] += \
            self.learning_rate * (self.df_value_state_map[next_state, 1] - self.df_value_state_map[current_state, 1])
        next_state = current_state
    else:
        current_state = df_all_episode_gameboards.iloc[num_rows - 1, 1].get_board_state_index()
        self.df_value_state_map.iloc[current_state, 1] = reward
        next_state = current_state

    for idx_row in range(2, num_rows + 1):
        # Update state value for 'Gameboard after player 2
        current_state = df_all_episode_gameboards.iloc[num_rows - idx_row, 2].get_board_state_index()
        next_state = df_all_episode_gameboards.iloc[num_rows - idx_row + 1, 1].get_board_state_index()
        self.df_value_state_map.iloc[current_state, 1] = \
            self.df_value_state_map.iloc[current_state, 1] + \
            self.learning_rate * (
                    self.df_value_state_map[next_state, 1] - self.df_value_state_map[current_state, 1])


np.savetxt("outfile.txt", write_array, fmt='%s')

for i in range(4,0):
    print(i)