import pygame

WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH//COLS

# rgb
LIGHTGREEN = (5,102,79)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREY = (145,145,145)

CROWN = pygame.transform.scale(pygame.image.load('checkers/crown.jpg'), (44, 25))
