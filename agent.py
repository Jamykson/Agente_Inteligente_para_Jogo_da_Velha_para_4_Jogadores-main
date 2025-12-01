import numpy as np
import pickle
import random
from settings import *

class QAgent:
    def __init__(self):
        self.q_table = {} # Dicionário: Chave="Estado Canônico", Valor=[Q-Values]
        self.epsilon = EPSILON_START
        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR

    def get_symmetry_info(self, board):
        """
        OTIMIZAÇÃO AVANÇADA: Estados Canônicos (Simetrias).
        
        Retorna:
        1. canonical_key: A string do tabuleiro na sua forma 'menor' (canônica).
        2. rotation: Quantas rotações de 90° (anti-horário) foram usadas.
        3. flip: Se houve espelhamento (flip) horizontal.
        
        Isso permite reduzir o espaço de estados em ~8x, pois ensina ao agente
        que um tabuleiro rotacionado é a mesma coisa que o original.
        """
        # 1. Generalização de Oponentes (Simplificação Visual)
        sim_board = board.copy()
        sim_board[sim_board == 3] = 2
        sim_board[sim_board == 4] = 2
        
        # 2. Busca pela forma canônica (Menor representação entre as 8 simetrias)
        symmetries = []
        
        # Gera candidatos: Original e suas rotações
        b = sim_board
        for r in range(4): # 0, 1, 2, 3 rotações
            # Adiciona a versão normal
            symmetries.append((tuple(b.flatten()), r, False))
            
            # Adiciona a versão espelhada (Flip Horizontal)
            b_flip = np.fliplr(b)
            symmetries.append((tuple(b_flip.flatten()), r, True))
            
            # Rotaciona o tabuleiro base para a próxima iteração
            b = np.rot90(b)
            
        # Escolhe a simetria que gerou a "menor" tupla (lexicograficamente)
        # Essa será a chave única para todas as 8 variações desse tabuleiro
        best_sym = min(symmetries, key=lambda x: x[0])
        
        return str(best_sym[0]), best_sym[1], best_sym[2]

    def map_action_to_canonical(self, action, rotation, flip):
        """
        Converte uma ação (índice 0-15) do Tabuleiro Real para o Tabuleiro Canônico.
        Necessário para buscar/atualizar o Q-Value correto na tabela.
        """
        row, col = divmod(action, BOARD_SIZE)
        
        # Aplica a mesma transformação geométrica que o tabuleiro sofreu
        
        # 1. Rotação (Anti-horária)
        for _ in range(rotation):
            # Fórmula da rotação 90 graus em matriz: (r, c) -> (N-1-c, r)
            row, col = BOARD_SIZE - 1 - col, row
            
        # 2. Espelhamento (Flip Horizontal)
        if flip:
            # Fórmula do flip: (r, c) -> (r, N-1-c)
            col = BOARD_SIZE - 1 - col
            
        return row * BOARD_SIZE + col

    def choose_action(self, board, valid_moves):
        """
        Escolhe a ação considerando as simetrias.
        """
        # Obtém a chave canônica e os dados de transformação
        state_key, rot, flip = self.get_symmetry_info(board)

        # 1. Exploração (Aleatório)
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # 2. Explotação (Inteligência)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)

        # Recupera os Q-Values da memória (que estão na orientação canônica)
        canonical_q_values = self.q_table[state_key]

        # Mapeia os Q-Values de volta para o Tabuleiro Real
        # Criamos um array de -Infinito para filtrar jogadas inválidas
        real_q_values = np.full(BOARD_SIZE * BOARD_SIZE, -np.inf)
        
        for move in valid_moves:
            # Descobre qual célula do tabuleiro canônico corresponde a este movimento real
            canon_move = self.map_action_to_canonical(move, rot, flip)
            
            # Atribui o valor aprendido ao movimento real
            real_q_values[move] = canonical_q_values[canon_move]
        
        # Retorna o índice da melhor ação no tabuleiro REAL
        return np.argmax(real_q_values)

    def learn(self, state, action, reward, next_state):
        """
        Atualiza a Q-Table mapeando as ações reais para as canônicas.
        """
        # Pega informações do estado atual (Real -> Canônico)
        state_key, rot, flip = self.get_symmetry_info(state)
        
        # Transforma a ação que foi feita no real para o índice correspondente no canônico
        canon_action = self.map_action_to_canonical(action, rot, flip)
        
        # Pega informações do próximo estado
        next_state_key, _, _ = self.get_symmetry_info(next_state)

        # Inicializa se necessário
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(BOARD_SIZE * BOARD_SIZE)

        # Bellman Equation
        old_value = self.q_table[state_key][canon_action]
        next_max = np.max(self.q_table[next_state_key]) # Max valor independe de rotação

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        
        # Atualiza a tabela na posição canônica correta
        self.q_table[state_key][canon_action] = new_value

    def save_model(self, filename="brain.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"💾 Modelo salvo em {filename} ({len(self.q_table)} estados canônicos).")

    def load_model(self, filename="brain.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"📂 Modelo carregado! Conhece {len(self.q_table)} padrões únicos.")
        except FileNotFoundError:
            print("⚠️ Arquivo não encontrado. Iniciando agente do zero.")