# settings.py
import numpy as np

BOARD_SIZE = 4
WIN_LENGTH = 3
NUM_PLAYERS = 4

EMPTY = 0
AGENT_ID = 1
OPPONENTS = [2, 3, 4]

SYMBOLS = {
    EMPTY: '.',
    AGENT_ID: 'X',
    2: 'O',
    3: '<',
    4: '^'
}

REWARDS = {
    'WIN': 500,        
    'LOSS': -300,      # <--- AUMENTADO: Perder deve ser ainda mais doloroso (-200 -> -300)
    'DRAW': -20,       
    'INVALID': -1000,  
    'STEP': -1,
    'THREAT': 2,       
    'BLOCK': 30,       # <--- NOVO: Recompensa por impedir uma vitória inimiga
    'IGNORE_DEFENSE': -100 
}

# Parâmetros de Treinamento
EPISODES = 600_000
DISCOUNT_FACTOR = 0.99

# Exploracao
EPSILON_START = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 0.999992 

# Aprendizado
ALPHA_START = 0.3      
ALPHA_MIN = 0.01       
ALPHA_DECAY = 0.999995