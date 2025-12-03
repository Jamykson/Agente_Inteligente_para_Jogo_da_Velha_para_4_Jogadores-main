import numpy as np
import pickle
import random
from settings import *

class QAgent:
    def __init__(self):
        self.q_table = {} 
        self.epsilon = EPSILON_START
        self.alpha = ALPHA_START # <--- Agora usa o valor inicial definido
        self.gamma = DISCOUNT_FACTOR

    def get_symmetry_info(self, board):
        """Retorna a chave canônica e as transformações necessárias."""
        sim_board = board.copy()
        sim_board[sim_board == 3] = 2
        sim_board[sim_board == 4] = 2
        
        symmetries = []
        b = sim_board
        for r in range(4): 
            symmetries.append((tuple(b.flatten()), r, False))
            b_flip = np.fliplr(b)
            symmetries.append((tuple(b_flip.flatten()), r, True))
            b = np.rot90(b)
            
        best_sym = min(symmetries, key=lambda x: x[0])
        return str(best_sym[0]), best_sym[1], best_sym[2]

    def map_action_to_canonical(self, action, rotation, flip):
        """Mapeia ação do mundo real para o canônico."""
        row, col = divmod(action, BOARD_SIZE)
        for _ in range(rotation):
            row, col = BOARD_SIZE - 1 - col, row
        if flip:
            col = BOARD_SIZE - 1 - col
        return row * BOARD_SIZE + col

    def choose_action(self, board, valid_moves):
        state_key, rot, flip = self.get_symmetry_info(board)

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)

        canonical_q_values = self.q_table[state_key]
        real_q_values = np.full(BOARD_SIZE * BOARD_SIZE, -np.inf)
        
        for move in valid_moves:
            canon_move = self.map_action_to_canonical(move, rot, flip)
            real_q_values[move] = canonical_q_values[canon_move]
        
        # OTIMIZAÇÃO: Quebra de empate aleatória
        # Se houver múltiplas melhores jogadas com o mesmo valor, escolhe uma ao acaso
        max_value = np.max(real_q_values)
        best_moves = [i for i, v in enumerate(real_q_values) if v == max_value]
        return random.choice(best_moves)

    def learn(self, state, action, reward, next_state):
        state_key, rot, flip = self.get_symmetry_info(state)
        canon_action = self.map_action_to_canonical(action, rot, flip)
        next_state_key, _, _ = self.get_symmetry_info(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)

        old_value = self.q_table[state_key][canon_action]
        next_max = np.max(self.q_table[next_state_key])

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_key][canon_action] = new_value

    def decay_alpha(self):
        """Reduz a taxa de aprendizado gradualmente."""
        if self.alpha > ALPHA_MIN:
            self.alpha *= ALPHA_DECAY

    def save_model(self, filename="brain.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"💾 Modelo salvo em {filename} ({len(self.q_table)} estados canônicos).")

    def load_model(self, filename="brain.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"📂 Modelo carregado! Alpha: {self.alpha:.4f}")
        except FileNotFoundError:
            print("⚠️ Arquivo não encontrado.")