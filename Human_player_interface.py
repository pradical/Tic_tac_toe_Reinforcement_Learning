from reinforcement_learning.Tic_tac_toe.GameBoardState import GameBoardState
import numpy as np
import re

class Human_player_interface:

    def __init__(self, player_mark : int):
        self.player_mark: int = player_mark
        assert player_mark in [1, 2]

    def choose_next_action_play(self, in_gameboard: GameBoardState) -> GameBoardState:

        action_taken = False

        while not action_taken:

            print('Please enter the move that you want to make.\n')
            print('Enter the \'(row, column)\' number of where you want to place your mark.\n')
            user_input = input('Example top left is (0,0).\n')

            pattern = r'\b(?:\({0,1}([0-2][ ,][0-2])\){0,1})\b'
            numbers = re.findall(pattern, user_input)

            if numbers:
                numbers = numbers[0]
                if ',' in numbers:
                    numbers = [int(x) for x in numbers.split(',')]
                else:
                    numbers = [int(x) for x in numbers.split(' ')]


                if in_gameboard.board_matrix[numbers[0], numbers[1]] != 0:
                    print('Please enter location which is empty. Invalid entry! Please try again.\n')
                    in_gameboard.draw_game_board()
                    continue

                in_gameboard.board_matrix[numbers[0], numbers[1]] = self.player_mark
                action_taken = True
                return in_gameboard

            else:
                print('Please enter valid location. Invalid entry! Please try again.\n')
                in_gameboard.draw_game_board()







