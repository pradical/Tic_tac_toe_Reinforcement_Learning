import numpy as np
from reinforcement_learning.Tic_tac_toe.convert_from_any_base_to_decimal import *


class GameBoardState:
    def __init__(self, in_array=np.full((3, 3), int(0))):
        # The board matrix is the literal representation of the board.
        # 0 indicates unmarked, 1 indicates player 1 mark, 2 indicates player 2 mark.
        self.board_matrix = in_array.copy()
        if np.all(self.board_matrix == 0):
            self.is_board_empty = True
        else:
            self.is_board_empty = False

    def get_board_state_index(self):
        # The index of the given state of board.
        # Every board has an associated state_index belonging to [0, 19682].
        # The index is simply the decimal conversion of the values of the board_matrix.
        # Used to assign value function of the state later.

        return toDeci(''.join([str(int(num)) for num in self.board_matrix.flatten()]), 3)

    def get_children_states(self):
        list_of_children_boards = []
        for i in range(self.board_matrix.shape[0]):
            for j in range(self.board_matrix.shape[1]):
                if self.board_matrix[i, j] != 0: continue
                new_state_matrix = self.board_matrix.copy()
                new_state_matrix[i, j] = 1
                list_of_children_boards.append(GameBoardState(in_array=new_state_matrix))

                new_state_matrix[i, j] = 2
                list_of_children_boards.append(GameBoardState(in_array=new_state_matrix))

        return list_of_children_boards

    def get_children_states_for_player(self, player_mark):
        assert player_mark in [1, 2]

        list_of_children_boards = []
        for i in range(self.board_matrix.shape[0]):
            for j in range(self.board_matrix.shape[1]):
                if self.board_matrix[i, j] != 0: continue
                new_state_matrix = self.board_matrix.copy()
                new_state_matrix[i, j] = player_mark
                list_of_children_boards.append(GameBoardState(in_array=new_state_matrix))

        return list_of_children_boards

    def update_board_state(self, in_array):
        self.board_matrix = in_array.copy()
        if np.all(self.board_matrix == 0):
            self.is_board_empty = True
        else:
            self.is_board_empty = False

    def is_board_end_state(self) -> bool:

        # Check if there are any marks: three in a row.
        for i in range(self.board_matrix.shape[0]):
            if (np.all(self.board_matrix[i, :] == self.board_matrix[i, 0])) \
                    and (self.board_matrix[i, 0] != 0): return True

        # Check if there are any marks: three in a column.
        for j in range(self.board_matrix.shape[1]):
            if np.all(self.board_matrix[:, j] == self.board_matrix[0, j]) \
                    and (self.board_matrix[0, j] != 0): return True

        # Check if there are any marks: three in any diagonal.
        if self.board_matrix[0, 0] == self.board_matrix[1, 1] == self.board_matrix[2, 2] != 0: return True
        if self.board_matrix[0, 2] == self.board_matrix[1, 1] == self.board_matrix[2, 0] != 0: return True

        # Check if the board is fully marked.
        if np.all(self.board_matrix != 0): return True

        return False

    def is_board_victory_state(self, player_mark: int) -> bool:

        # Check if there are any marks: three in a row.
        for i in range(self.board_matrix.shape[0]):
            if (np.all(self.board_matrix[i, :] == self.board_matrix[i, 0])) \
                    and (self.board_matrix[i, 0] == player_mark): return True

        # Check if there are any marks: three in a column.
        for j in range(self.board_matrix.shape[1]):
            if np.all(self.board_matrix[:, j] == self.board_matrix[0, j]) \
                    and (self.board_matrix[0, j] == player_mark): return True

        # Check if there are any marks: three in any diagonal.
        if self.board_matrix[0, 0] == self.board_matrix[1, 1] == self.board_matrix[2, 2] == player_mark: return True
        if self.board_matrix[0, 2] == self.board_matrix[1, 1] == self.board_matrix[2, 0] == player_mark: return True

        return False

    def is_board_defeat_state(self, player_mark: int) -> bool:

        # Check if there are any marks: three in a row.
        for i in range(self.board_matrix.shape[0]):
            if (np.all(self.board_matrix[i, :] == self.board_matrix[i, 0])) \
                    and (self.board_matrix[i, 0] == (3 - player_mark)):
                return True  # (3 - player_mark returns other players mark)

        # Check if there are any marks: three in a column.
        for j in range(self.board_matrix.shape[1]):
            if np.all(self.board_matrix[:, j] == self.board_matrix[0, j]) \
                    and (self.board_matrix[0, j] == (3 - player_mark)): return True

        # Check if there are any marks: three in any diagonal.
        if self.board_matrix[0, 0] == self.board_matrix[1, 1] == self.board_matrix[2, 2] == (3 - player_mark):
            return True
        if self.board_matrix[0, 2] == self.board_matrix[1, 1] == self.board_matrix[2, 0] == (3 - player_mark):
            return True

        return False

    def draw_game_board(self):

        # This function prints out the board that it was passed.
        # "board" is a list of 10 strings representing the board (ignore index 0)
        board = self.board_matrix.ravel().astype(str)
        board = np.where(board == '1', 'X', board)
        board = np.where(board == '2', 'O', board)
        board = np.where(board == '0', ' ', board)
        board = board.reshape(-1,3)
        print('   |   |')
        print(' ' + str(board[0, 0]) + ' | ' + str(board[0, 1]) + ' | ' + str(
            board[0, 2]))
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + str(board[1, 0]) + ' | ' + str(board[1, 1]) + ' | ' + str(
            board[1, 2]))
        print('   |   |')
        print('-----------')
        print('   |   |')
        print(' ' + str(board[2, 0]) + ' | ' + str(board[2, 1]) + ' | ' + str(
            board[2, 2]))
        print('   |   |')


if __name__ == '__main__':
    yeno = GameBoardState()
    yeno.board_matrix = np.array([[1, 0, 1],
                                  [2, 1, 1],
                                  [2, 2, 0]])

    for whatever in yeno.get_children_states():
        print(whatever.board_matrix)
