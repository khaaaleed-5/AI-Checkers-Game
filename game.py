import numpy as np
import pygame
from checkers.constants import RED, WHITE, BLUE, SQUARE_SIZE
from checkers.board import Board
from minimax import minimax
from checkers.piece import Piece
from model import NeuralNetwork

class Game:
    def __init__(self, win):
        self._init()
        self.win = win
        self.model = NeuralNetwork()
    
    def update(self):
        self.board.draw(self.win)
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()

    def _init(self):
        self.selected = None
        self.board = Board()
        self.piece = Piece()
        self.turn = RED
        self.valid_moves = {}

    def winner(self):
        return self.board.winner()
    
    def get_board(self):
        return self.board
    
    def get_board_matrix(self):
        """
        Returns the current state of the board as a matrix.
        """
        return self.board.get_board_matrix()
    

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
    
    #for genetic
    def execute_move(self,action):
        row,col,new_row,new_col = action[0][:4]
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
    
    #initialize no of boards
    def init_board(self,population:list):
        boards = []
        for model in population:
            boards.append({'x':self.get_board_matrix(),'fitness':0})
        return boards

    def evaluate_population(self, population: list):
        '''
        fitness for model is divided:
           1. - 1 for piece   ->after game
           2. - 1 if killed white piece (12 - pieces) ->after game
           3. - 2 for king    ->during game
           4.- -1 if predicted illegal move ->during game
           5.- 50 if model won (not now)  ->during game
           
        '''
        while True:

            fitness_scores = []
            for elm in population:
                while True:
                    '''
                    break if:
                        - 
                    '''
                    #minimax
                    if self.turn == WHITE:
                        _, new_board = minimax(self.get_board(), 4, WHITE, self)
                        self.minimax_move(new_board)
                    
                    #model
                    else:
                        pred = ...   #model prediction
                        x = np.argmax(pred)
                        index_number = x%4
                        index_list = x // len(pred)
                        if not self.piece.king and index_number > 2:
                            fitness_scores.append(-1)
                            break
                        elif

                    
                    board = self.get_board()
                    for row in board:
                        for piece in row:


                fitness_scores = np.array([board['fitness'] for board in boards])
                    
            return fitness_scores