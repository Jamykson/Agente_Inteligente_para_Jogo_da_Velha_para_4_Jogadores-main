# environment.py (Apenas o método step precisa ser substituído, mas envio o arquivo ajustado)
import numpy as np
import random
from settings import *

class TicTacToeEnv:
    def __init__(self):
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

        # Linhas e Colunas
        for i in range(n):
            row = b[i, :]
            col = b[:, i]
            for j in range(n - target + 1):
                if np.all(row[j:j+target] == player_id): return True
                if np.all(col[j:j+target] == player_id): return True

        # Diagonais
        for r in range(n - target + 1):
            for c in range(n - target + 1):
                subgrid = b[r:r+target, c:c+target]
                if np.all(subgrid.diagonal() == player_id): return True
                if np.all(np.fliplr(subgrid).diagonal() == player_id): return True

        return False

    def count_threats(self, player_id):
        count = 0
        b = self.board
        n = BOARD_SIZE
        w = WIN_LENGTH

        def check_window(window):
            pieces = np.count_nonzero(window == player_id)
            empties = np.count_nonzero(window == EMPTY)
            return pieces == (w - 1) and empties == 1

        for i in range(n):
            for j in range(n - w + 1):
                if check_window(b[i, j:j+w]): count += 1
                if check_window(b[j:j+w, i]): count += 1

        for r in range(n - w + 1):
            for c in range(n - w + 1):
                sub = b[r:r+w, c:c+w]
                if check_window(sub.diagonal()): count += 1
                if check_window(np.fliplr(sub).diagonal()): count += 1
        return count

    def get_dangerous_cells(self):
        dangerous_cells = []
        empty_cells = [i for i in range(BOARD_SIZE * BOARD_SIZE) if self.is_valid_move(i)]
        
        for cell in empty_cells:
            row, col = divmod(cell, BOARD_SIZE)
            for opp in OPPONENTS:
                self.board[row, col] = opp 
                if self.check_winner(opp):
                    dangerous_cells.append(cell)
                self.board[row, col] = EMPTY 
                if cell in dangerous_cells: break
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

        # 1. Validade
        if not self.is_valid_move(action):
            return self.board.flatten(), REWARDS['INVALID'], self.done, {}

        # 2. Análise de Defesa (Antes de jogar)
        dangerous_cells = self.get_dangerous_cells()
        is_successful_block = False
        ignored_defense = False
        
        if len(dangerous_cells) > 0:
            if action in dangerous_cells:
                is_successful_block = True # <--- Bloqueou com sucesso!
            else:
                ignored_defense = True     # <--- Ignorou o perigo!

        # 3. Executa Jogada
        row, col = divmod(action, BOARD_SIZE)
        threats_before = self.count_threats(AGENT_ID)
        self.board[row, col] = AGENT_ID

        # 4. Verifica Vitória
        if self.check_winner(AGENT_ID):
            self.done = True
            return self.board.flatten(), REWARDS['WIN'], True, {'result': 'Win'}

        # 5. Calcula Recompensa
        reward = REWARDS['STEP']
        
        # Bônus de Ataque
        threats_after = self.count_threats(AGENT_ID)
        if threats_after > threats_before:
            reward += REWARDS['THREAT']

        # Bônus de Defesa (NOVO)
        if is_successful_block:
            reward += REWARDS['BLOCK']
        
        # Punição de Defesa
        if ignored_defense:
            reward += REWARDS['IGNORE_DEFENSE']

        # Empate?
        if self.is_draw():
            self.done = True
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        # Oponentes jogam
        self.play_opponents()

        if self.done:
            if self.winner in OPPONENTS:
                return self.board.flatten(), REWARDS['LOSS'], True, {'result': 'Loss'}
            return self.board.flatten(), REWARDS['DRAW'], True, {'result': 'Draw'}

        return self.board.flatten(), reward, False, {}