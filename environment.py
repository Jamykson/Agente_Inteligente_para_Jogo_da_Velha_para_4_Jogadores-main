import numpy as np
import random
from settings import *

class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia o tabuleiro para o estado vazio."""
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        self.winner = None
        return self.board.flatten() # Retorna o estado inicial

    def is_valid_move(self, action):
        """Verifica se a célula está dentro do grid e vazia."""
        row, col = divmod(action, BOARD_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return self.board[row, col] == EMPTY
        return False

    def check_winner(self, player_id):
        """
        Verifica vitória com Janela Deslizante.
        """
        b = self.board
        n = BOARD_SIZE
        target = WIN_LENGTH # 3

        # 1. Verificar Linhas e Colunas
        for i in range(n):
            row = b[i, :]
            col = b[:, i]
            for j in range(n - target + 1):
                if np.all(row[j:j+target] == player_id): return True
                if np.all(col[j:j+target] == player_id): return True

        # 2. Verificar Diagonais
        for r in range(n - target + 1):
            for c in range(n - target + 1):
                subgrid = b[r:r+target, c:c+target]
                if np.all(subgrid.diagonal() == player_id): return True
                if np.all(np.fliplr(subgrid).diagonal() == player_id): return True

        return False

    def count_threats(self, player_id):
        """
        Conta quantas sequências de 2 peças o jogador tem
        que ainda podem virar vitória (tem espaço vazio).
        """
        count = 0
        b = self.board
        n = BOARD_SIZE
        w = WIN_LENGTH # 3

        # Helper interno
        def check_window(window):
            pieces = np.count_nonzero(window == player_id)
            empties = np.count_nonzero(window == EMPTY)
            return pieces == (w - 1) and empties == 1

        # Linhas e Colunas
        for i in range(n):
            for j in range(n - w + 1):
                if check_window(b[i, j:j+w]): count += 1
                if check_window(b[j:j+w, i]): count += 1

        # Diagonais
        for r in range(n - w + 1):
            for c in range(n - w + 1):
                sub = b[r:r+w, c:c+w]
                if check_window(sub.diagonal()): count += 1
                if check_window(np.fliplr(sub).diagonal()): count += 1
        
        return count

    def get_dangerous_cells(self):
        """
        Identifica células onde, se um oponente jogar, ele ganha.
        Retorna uma lista de índices dessas células.
        """
        dangerous_cells = []
        empty_cells = [i for i in range(BOARD_SIZE * BOARD_SIZE) if self.is_valid_move(i)]
        
        for cell in empty_cells:
            row, col = divmod(cell, BOARD_SIZE)
            # Simula cada oponente jogando nessa célula
            for opp in OPPONENTS:
                self.board[row, col] = opp # Simula
                if self.check_winner(opp):
                    dangerous_cells.append(cell)
                self.board[row, col] = EMPTY # Desfaz
                
                if cell in dangerous_cells:
                    break
                    
        return dangerous_cells

    def is_draw(self):
        return not np.any(self.board == EMPTY)

    def play_opponents(self):
        if self.done: return

        for opp_id in OPPONENTS:
            empty_cells = np.argwhere(self.board == EMPTY)
            if len(empty_cells) == 0:
                self.done = True
                return

            choice = random.choice(empty_cells)
            self.board[choice[0], choice[1]] = opp_id

            if self.check_winner(opp_id):
                self.winner = opp_id
                self.done = True
                return
            
            if self.is_draw():
                self.done = True
                return

    def step(self, action):
        if self.done: return self.board.flatten(), 0, True, {}

        # 1. Checa validade
        if not self.is_valid_move(action):
            return self.board.flatten(), REWARDS['INVALID'], self.done, {}

        # --- LÓGICA NOVA DE DEFESA ---
        dangerous_cells = self.get_dangerous_cells()
        ignored_defense = False
        
        # Se existiam células perigosas e o agente NÃO jogou nelas...
        if len(dangerous_cells) > 0 and action not in dangerous_cells:
            ignored_defense = True
        # -----------------------------

        # Faz a jogada
        row, col = divmod(action, BOARD_SIZE)
        threats_before = self.count_threats(AGENT_ID) # Conta ameaças ANTES
        self.board[row, col] = AGENT_ID

        # Verifica Vitória
        if self.check_winner(AGENT_ID):
            self.done = True
            return self.board.flatten(), REWARDS['WIN'], True, {'result': 'Win'}

        # Calcula Recompensa
        reward = REWARDS['STEP']
        
        # Bônus de Ataque (Threats)
        threats_after = self.count_threats(AGENT_ID) # Conta ameaças DEPOIS
        if threats_after > threats_before:
            reward += REWARDS['THREAT']

        # Punição de Defesa (NOVO)
        if ignored_defense:
            reward += REWARDS['IGNORE_DEFENSE']

        # Verifica Empate
        if self.is_draw():
            self.done = True
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        # Turno Oponentes
        self.play_opponents()

        if self.done:
            if self.winner in OPPONENTS:
                return self.board.flatten(), REWARDS['LOSS'], True, {'result': 'Loss'}
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        return self.board.flatten(), reward, False, {}