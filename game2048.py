import numpy as np
from numpy import ndarray
import pygame
import time

# Define board size
BOARD_SIZE = 4

# Define tile size
TILE_SIZE = 100

# Define font size
FONT_SIZE = 40

# Set the size of the game window
WINDOW_SIZE = (BOARD_SIZE * TILE_SIZE, BOARD_SIZE * TILE_SIZE + FONT_SIZE)

tile_colors = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (62, 57, 51),
    8192: (62, 57, 51),
    16384: (62, 57, 51),
    32768: (62, 57, 51),
    65536: (62, 57, 51),
}


def stack(board):
    for i in range(0, len(board)):
        for j in range(0, len(board)):
            k = i

            while board[k][j] == 0:
                if k == len(board) - 1:
                    break
                k += 1

            if k != i:
                board[i][j], board[k][j] = board[k][j], 0


def sum_up(board):
    for i in range(0, len(board) - 1):
        for j in range(0, len(board)):
            if board[i][j] != 0 and board[i][j] == board[i + 1][j]:
                board[i][j] += board[i + 1][j]
                board[i + 1][j] = 0


class Board:
    def __init__(self, size):
        self.__board = np.zeros((size, size))
        self.__score = 0

    def get_board(self) -> ndarray:
        return self.__board

    def get_val(self, i, j):
        return self.__board[i][j]

    def neighbour_values(self, i, j) -> list:
        x = [-1, 1,  0, 0]
        y = [ 0, 0, -1, 1]
        w = len(self.__board)
        vals = []

        for ind in range(0, len(x)):
            nx = i + x[ind]
            ny = j + y[ind]

            if nx < 0 or nx >= w or ny < 0 or ny >= w:
                continue

            vals.append(self.__board[nx][ny])

        return vals

    def completed(self):
        empty_tiles_count = len(np.where(self.__board == 0)[0])
        can_simplify_any = False
        w = len(self.__board)

        for i in range(0, w):
            should_break = False

            for j in range(0, w):
                if self.__board[i][j] in self.neighbour_values(i, j):
                    can_simplify_any = True
                    should_break = True
                    break

            if should_break:
                break

        return empty_tiles_count == 0 and not can_simplify_any

    def move(self, action, should_paint=False):
        self.drop_elem()

        if should_paint:
            self.paint()

        rotated_board = np.rot90(self.__board, action)
        stack(rotated_board)
        sum_up(rotated_board)
        stack(rotated_board)
        self.__board = np.rot90(rotated_board, len(self.__board) - action)

        self.calculate_score()

    def calculate_score(self):
        self.__score = np.max(self.__board)

    def paint(self):
        print(np.array_str(self.__board))

    def score(self):
        return self.__score

    def drop_elem(self):
        zeroes_flatten = np.where(self.__board == 0)
        zeroes_indices = [(x, y) for x, y in zip(zeroes_flatten[0], zeroes_flatten[1])]

        if len(zeroes_indices) == 0:
            return

        random_index = zeroes_indices[0]
        self.__board[random_index] = 2



def board_fitness(board: Board, i):
    board_length = len(board.get_board())
    score = 0
    used_tiles = np.count_nonzero(board.get_board() > 0)

    for i in range(0, board_length):
        for j in range(0, board_length):
            score += board.get_val(i, j)

    return score / used_tiles

    # if i == 0:
    #     return score + 2
    # elif i == 2:
    #     return score - 2
    # else:
    #     return score + 1

class Solver:
    def __init__(self):
        self.board = Board(4)
        self.steps = []




class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("2048")
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.board = Board(4)
        self.steps = []

    def run(self, steps: ndarray):
        while steps.any():
            pygame.event.get()

            curr_move = steps[0]
            steps = steps[1:]

            self.board.move(curr_move, True)
            self.paint(self.board.get_board())

            if self.board.completed():
                break

            time.sleep(0.25)

        # self.paint(self.board.get_board())
        #
        # while True:
        #     pygame.event.get()
        #
        #     best_board_score = -1
        #     best_board_action = -1
        #
        #     indeces = [1, 3, 0, 2]
        #
        #     for i in indeces:
        #         new_board = self.board
        #         new_board.move(i)
        #         fitness = board_fitness(new_board, i)
        #
        #         if fitness > best_board_score:
        #             best_board_score = fitness
        #             best_board_action = i
        #
        #     self.board.move(best_board_action, True)
        #     self.paint(self.board.get_board())
        #     self.steps.append(best_board_action)
        #
        #     if self.board.completed():
        #         break
        #
        #     time.sleep(0.25)
        #
        # input()

    def paint(self, board):
        block_size = 100
        font_size = 50

        for i in range(len(board)):
            for j in range(len(board)):
                value = int(board[i][j])
                color = tile_colors[value]
                rect = pygame.Rect(j * block_size, i * block_size, block_size, block_size)
                pygame.draw.rect(self.screen, color, rect)

                if value != 0:
                    font = pygame.font.Font(None, font_size)
                    text = font.render(str(value), True, (255, 255, 255))
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)

        pygame.display.update()
