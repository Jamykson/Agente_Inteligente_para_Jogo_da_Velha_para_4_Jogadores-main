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
    'WIN': 100,        # <--- ALTERADO: Aumentado para 100 (Prioridade máxima)
    'LOSS': -100,      # Mantido: Perder é inaceitável
    'DRAW': -10,       # <--- AJUSTADO: -2 era pouco. Empate deve doer mais que um passo normal.
    'INVALID': -200,   # Proibido errar casa
    'STEP': -1,
    'THREAT': 2,       # <--- AJUSTADO: Reduzido de 5 para 2. Evita que ele "enrole" criando ameaças em vez de ganhar.
    'IGNORE_DEFENSE': -50 # <--- AJUSTADO: Punição mais severa para ignorar bloqueios.
}

# Parâmetros de Treinamento
EPISODES = 600_000      # <--- AUMENTADO: Necessário para o agente aprender todas as simetrias perfeitamente.
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99  # Visão de longo prazo
EPSILON_START = 1.0
EPSILON_MIN = 0.001
EPSILON_DECAY = 0.999992 # <--- AJUSTADO: Decaimento mais lento para cobrir os 600k episódios.