import numpy as np
import pygame
from checkers.constants import RED, WHITE, BLUE, SQUARE_SIZE
from checkers.board import Board
from minimax import minimax
from checkers.piece import Piece
from model import Player

class Game:
    def __init__(self, win):
        self._init()
        self.win = win
        self.model = Player()
        
    def _init(self):
        self.selected = None
        self.board = Board()
        # self.piece = Piece()
        self.turn = RED
        self.valid_moves = {}
        self.piece_map = self.map_pieces()
    
    def init_board(self,population:list):
        boards = []
        for model in population:
            boards.append({'x':self.get_board_matrix(),'fitness':0})
        return boards
    
    def map_pieces(self):
        board = self.get_board_matrix()
        piece = 1
        map = {}
        for i in range(len(board), -1, -1):
            for j in range(len(board[i])):
                if board[i][j] in [2, 4]:
                    map[piece] = (i, j)
                    piece += 1
        return map
    
    def update(self):
        self.board.draw(self.win)
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()

    def winner(self):
        return self.board.winner()
    
    def get_board(self):
        return self.board
    
    def get_board_matrix(self):
        """
        Returns the current state of the board as a matrix.
        """
        return self.board.get_board_matrix()
    
    def get_piece_row_col(self, piece_id):
        return self.piece_map[piece_id]
    
    def get_new_row_col(self, row, col, move_id):

        if move_id == 1:
            return row + 1, col - 1
        elif move_id == 2:
            return row + 1, col + 1
        elif move_id == 3:
            return row - 1, col - 1
        elif move_id == 4:
            return row - 1, col + 1
        else:
            return row, col
    
    def reset(self):
        self._init()

    def select(self, row, col):
        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)
        
        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            return True
            
        return False

    def _move(self, row, col):
        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
            self.change_turn()
        else:
            return False

        return True
    
    def execute_move(self,row,col,new_row,new_col):
        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move(self.selected, new_row, new_col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
            self.change_turn()
        else:
            return False
        return True
    
    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            pygame.draw.circle(self.win, BLUE, (col * SQUARE_SIZE + SQUARE_SIZE//2, row * SQUARE_SIZE + SQUARE_SIZE//2), 15)

    def change_turn(self):
        self.valid_moves = {}
        if self.turn == RED:
            self.turn = WHITE
        else:
            self.turn = RED
    
    def minimax_move(self,board):
        self.board = board
        self.change_turn()
    
    def get_score(self):
        board = self.get_board_matrix()
        player1_pieces = player1_kings = player2_pieces = player2_kings = 0
        for row in board:
            for val in row:
                if val == 0:
                    continue
                else:
                    if val == 1:
                        player1_pieces += 1
                    elif val == 2:
                        player2_pieces += 1
                    elif val == 3:
                        player1_kings += 1
                    else:
                        player2_kings += 1
        # Calculate the score based on the number of pieces and kings
        score = (player1_pieces + 2 * player1_kings) - (player2_pieces + 2 * player2_kings)
        return score

    def evaluate_population(self, population: list[Player]):
        '''
        fitness for model is divided:
            1. - 1 for piece   ->after game
            2. - 1 if killed white piece (12 - pieces) ->after game
            3. - 2 for king    ->after game
            4.- -1 if predicted illegal move ->during game
            5.- 50 if model won (not now)  ->during game
        '''
        running = True
        clock = pygame.time.Clock()
        while running:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            fitness_scores = []
            for player in population:
                while True: # Play the game
                    if self.turn == WHITE: # Minimax's turn
                        _, new_board = minimax(self.get_board(), 4, WHITE, self)
                        self.minimax_move(new_board)

                    else: # Model's turn
                        pred = player.forward(self.get_board())
                        x = np.argmax(pred)
                        move_id = x % 4
                        piece_id = x // len(pred)

                        # get the piece row and col
                        row, col = self.get_piece_row_col(piece_id)
                        new_row, new_col = self.get_new_row_col(row, col, move_id)

                        # Check if the move is invalid
                        if not self.piece.king and move_id > 2:
                            fitness_scores.append(-1)
                            break

                        # Check if move is invalid
                        if self.board.get_valid_moves({'row': row, 'col': col}) == {}: # No valid moves
                            fitness_scores.append(-1)
                            break

                        # Execute the move
                        self.execute_move(row, col, new_row, new_col)
                    
                    # Check if the game is over
                    if self.winner() != None:
                        if self.winner() == RED:
                            fitness_scores.append(50)
                        break

                    # Update the board
                    self.update()
                
                # Calculate the fitness score
                model_score = 0
                for row in self.board:
                    for piece in row:
                        if piece == RED:
                            if piece.king:
                                model_score += 2
                            else:
                                model_score += 1
                        elif piece == WHITE:
                            if piece.king:
                                model_score -= 2
                            else:
                                model_score -= 1

                fitness_scores.append(model_score)

        return np.array(fitness_scores)


"""
[[0, 1, 0, 1, 0, 1, 0, 1],
[1, 0, 1, 0, 1, 0, 1, 0],
[0, 1, 0, 1, 0, 1, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[2, 0, 2, 0, 2, 0, 2, 0],
[0, 2, 0, 2, 0, 2, 0, 2],
[2, 0, 2, 0, 2, 0, 2, 0]]
"""