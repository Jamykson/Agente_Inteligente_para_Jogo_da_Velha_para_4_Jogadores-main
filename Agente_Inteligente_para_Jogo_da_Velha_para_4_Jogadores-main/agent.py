import numpy as np
import pickle
import random
from settings import *

class QAgent:
    def __init__(self):
        self.q_table = {} # Dicionário: Chave="Estado", Valor=[Lista de Q-Values]
        self.epsilon = EPSILON_START
        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR

    def get_state_key(self, board):
        """
        OTIMIZAÇÃO: Generalização de Oponentes.
        Transforma o tabuleiro para que todos os oponentes (2, 3, 4)
        pareçam o mesmo número (ex: 2).
        Isso reduz drasticamente o espaço de estados e acelera o aprendizado.
        """
        # Cria uma cópia para não alterar o jogo real
        simplified_board = board.copy()
        
        # Substitui 3 e 4 por 2. 
        # Agora o Agente vê: 0=Vazio, 1=Eu, 2=Inimigo (Qualquer um)
        simplified_board[simplified_board == 3] = 2
        simplified_board[simplified_board == 4] = 2
        
        return str(simplified_board.flatten())

    def choose_action(self, board, valid_moves):
        """
        Algoritmo Epsilon-Greedy:
        - Com chance 'epsilon': Escolhe aleatório (Exploração)
        - Caso contrário: Escolhe a ação com maior Q-Value (Explotação)
        """
        state_key = self.get_state_key(board)

        # 1. Exploração (Aleatório)
        # Se o dado cair num número baixo, ele chuta uma posição válida qualquer
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # 2. Explotação (Inteligência)
        # Se nunca viu esse estado, inicializa com zeros na memória
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)

        # Pega os valores Q conhecidos para este estado
        q_values = self.q_table[state_key]

        # Filtro de Segurança:
        # Criamos uma lista temporária onde jogadas inválidas têm valor -Inifinito
        # Isso garante que a IA nunca escolha jogar numa casa ocupada quando estiver jogando sério
        masked_q_values = np.full(BOARD_SIZE * BOARD_SIZE, -np.inf)
        
        for move in valid_moves:
            masked_q_values[move] = q_values[move]
        
        # Retorna o índice da ação com maior valor
        return np.argmax(masked_q_values)

    def learn(self, state, action, reward, next_state):
        """
        Atualiza a Q-Table usando a Equação de Bellman.
        Q_novo = Q_velho + alpha * [Recompensa + gamma * max(Q_futuro) - Q_velho]
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Garante que os estados existam na tabela antes de atualizar
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)

        # Valores para o cálculo
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key]) # O melhor valor possível do próximo estado

        # A Fórmula Mágica
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        # Grava o novo conhecimento
        self.q_table[state_key][action] = new_value

    def save_model(self, filename="brain.pkl"):
        """Salva o cérebro treinado em um arquivo."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"💾 Modelo salvo em {filename} ({len(self.q_table)} estados aprendidos).")

    def load_model(self, filename="brain.pkl"):
        """Carrega um cérebro treinado."""
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"📂 Modelo carregado! Conhece {len(self.q_table)} situações.")
        except FileNotFoundError:
            print("⚠️ Arquivo não encontrado. Iniciando agente do zero.")