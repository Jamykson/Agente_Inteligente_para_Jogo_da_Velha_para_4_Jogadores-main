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
    'WIN': 500,        # Ganhar
    'LOSS': -300,      # Perder
    'DRAW': -20,       # Empate
    'INVALID': -1000,  # Movimento Inválido
    'STEP': -1,        # Penalidade por perda de tempo
    'THREAT': 2,       # Recompensa por criar uma ameaça
    'BLOCK': 30,       # Recompensa por impedir uma vitória inimiga
    'IGNORE_DEFENSE': -100 # Inimigo tinha chance de ganhar e o agente ignorou
}

# Parâmetros de Treinamento
EPISODES = 600_000 #numero de repeticoes
DISCOUNT_FACTOR = 0.99 #importancia à vitórias futuras

# Exploracao Para ele ir diminuindo a aleatoriedade
EPSILON_START = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 0.999992 

# Aprendizado, ele vai substituindo cada vez menos o conhecimento antigo pelo novo
ALPHA_START = 0.3      
ALPHA_MIN = 0.01       
ALPHA_DECAY = 0.999995