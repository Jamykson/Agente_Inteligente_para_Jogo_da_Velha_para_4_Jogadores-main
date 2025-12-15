# environment.py
import numpy as np
import random
from settings import *

class TicTacToeEnv:
    def __init__(self, opponent_brains=None):
        """
        opponent_brains: Dicionário {player_id: brain_instance}.
        Exemplo: {2: elite_brain, 3: basic_brain, 4: None}
        Se a chave não existir ou for None, joga Aleatório.
        """
        # ATUALIZAÇÃO: Aceita um dicionário, não apenas um único cérebro
        self.opponent_brains = opponent_brains if opponent_brains else {}
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        self.winner = None
        return self.board.flatten()

    def is_valid_move(self, action):
        row, col = divmod(action, BOARD_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row, col] == EMPTY
        return False

    def check_winner(self, player_id):
        b = self.board
        n = BOARD_SIZE
        target = WIN_LENGTH

        # 1. Checar Linhas e Colunas
        for i in range(n):
            row = b[i, :]
            col = b[:, i]
            for j in range(n - target + 1):
                if np.all(row[j : j + target] == player_id): return True
                if np.all(col[j : j + target] == player_id): return True

        # 2. Checar Diagonais
        for r in range(n - target + 1):
            for c in range(n - target + 1):
                sub_grid = b[r : r + target, c : c + target]
                if np.all(sub_grid.diagonal() == player_id): return True
                if np.all(np.fliplr(sub_grid).diagonal() == player_id): return True
                    
        return False

    def is_draw(self):
        return not np.any(self.board == EMPTY)

    def get_opponent_view(self, player_id):
        """
        Gera a ilusão de ótica: O oponente 'player_id' vê o tabuleiro 
        como se ELE fosse o Jogador 1 e os outros fossem inimigos.
        """
        view = self.board.copy()
        
        # 1. Identificar onde é o jogador atual
        my_pos = (view == player_id)
        
        # 2. Todo o resto que não é vazio vira '2' (Inimigo Genérico)
        others_pos = (view != 0) & (view != player_id)
        view[others_pos] = 2 
        
        # 3. Transformar o jogador atual em '1' (Para bater com a Q-Table)
        view[my_pos] = 1
        
        return view # Retorna matriz 4x4

    def play_opponents(self):
        if self.done: return
        
        for opp_id in OPPONENTS:
            # Checa empate antes de cada movimento
            if self.is_draw(): 
                self.done = True
                return

            valid_moves = [i for i in range(BOARD_SIZE**2) if self.board.flatten()[i] == 0]
            if not valid_moves: 
                self.done = True
                return

            # --- LÓGICA DO BATTLE ROYALE ---
            # Verifica se existe um cérebro específico para este ID na mesa
            brain = self.opponent_brains.get(opp_id)
            
            if brain:
                # Se tem cérebro, usa a visão inteligente
                opp_view = self.get_opponent_view(opp_id)
                
                # Garante que ele jogue sério (sem errar de propósito)
                old_eps = brain.epsilon
                brain.epsilon = 0.0
                action = brain.choose_action(opp_view, valid_moves)
                brain.epsilon = old_eps # Restaura configuração original
            else:
                # Se for None (ou não estiver no dict), joga o CAOS (Aleatório)
                action = random.choice(valid_moves)
            # -------------------------------

            row, col = divmod(action, BOARD_SIZE)
            self.board[row, col] = opp_id

            if self.check_winner(opp_id):
                self.winner = opp_id
                self.done = True
                return
            
            if self.is_draw(): 
                self.done = True
                return

    def step(self, action):
        if self.done: return self.board.flatten(), 0, True, {}
        
        # 1. Validade
        if not self.is_valid_move(action):
            return self.board.flatten(), REWARDS['INVALID'], self.done, {}

        # 2. Executa Jogada do Agente Principal
        row, col = divmod(action, BOARD_SIZE)
        self.board[row, col] = AGENT_ID
        
        if self.check_winner(AGENT_ID):
            self.done = True
            return self.board.flatten(), REWARDS['WIN'], True, {'result': 'Win'}
            
        reward = REWARDS['STEP']
        if self.is_draw():
            self.done = True
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        # 3. Oponentes jogam (Usando lógica mista definida no init)
        self.play_opponents()

        if self.done:
            if self.winner in OPPONENTS:
                return self.board.flatten(), REWARDS['LOSS'], True, {'result': 'Loss'}
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        return self.board.flatten(), reward, False, {}