# settings.py (Atualizado)
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
    'WIN': 50,         # Aumentei muito: Ganhar é tudo.
    'LOSS': -100,      # Perder é inaceitável (Dobrei a punição)
    'DRAW': -2,        # Empate é levemente ruim
    'INVALID': -200,   # Proibido errar casa
    'STEP': -1,
    'THREAT': 5,        # Aumentei o incentivo para criar jogadas de ataque
    'IGNORE_DEFENSE': -40
}

EPISODES = 400_000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99  # Aumentei para 0.99. Ele vai pensar MUITO no futuro.
EPSILON_START = 1.0
EPSILON_MIN = 0.001     # <--- O SEGREDO: Reduzir o erro aleatório final
EPSILON_DECAY = 0.99999 # Decaimento super lento para cobrir os 400k episódios