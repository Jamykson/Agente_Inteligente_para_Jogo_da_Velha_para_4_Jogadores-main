import numpy as np
import time
from environment import TicTacToeEnv
from agent import QAgent
from settings import *

def train():
    env = TicTacToeEnv()
    agent = QAgent()
    
    print(f"🚀 Iniciando Treinamento: {EPISODES} episódios.")
    print(f"Campo: {BOARD_SIZE}x{BOARD_SIZE} | Vitória: {WIN_LENGTH} em linha")
    print("-" * 50)
    
    start_time = time.time()
    recent_wins = [] # Para calcular média móvel
    
    for episode in range(1, EPISODES + 1):
        # 1. Resetar Jogo
        state = env.reset() # Recebe o tabuleiro 4x4 achatado
        state_matrix = env.board # Mantemos a referência da matriz para passar ao agente
        done = False
        
        while not done:
            # Descobre quais casas estão vazias (para acelerar o aprendizado)
            valid_moves = [i for i in range(BOARD_SIZE * BOARD_SIZE) if env.is_valid_move(i)]
            
            # Se não tem movimento válido, é empate técnico
            if not valid_moves:
                break

            # 2. Agente escolhe ação
            action = agent.choose_action(state_matrix, valid_moves)
            
            # 3. Ambiente processa (Agente joga -> Oponentes jogam)
            next_state_flat, reward, done, info = env.step(action)
            next_state_matrix = env.board # Pega a matriz atualizada

            # 4. Agente aprende com o resultado
            agent.learn(state_matrix, action, reward, next_state_matrix)
            
            # Atualiza estado atual
            state_matrix = next_state_matrix.copy()

        # --- Fim do Episódio ---
        
        # Registra estatística (1 se ganhou, 0 se perdeu/empatou)
        if info.get('result') == 'Win':
            recent_wins.append(1)
        else:
            recent_wins.append(0)
            
        # Mantém histórico dos últimos 1000 jogos apenas
        if len(recent_wins) > 1000:
            recent_wins.pop(0)

        # Decaimento do Epsilon (Diminuir a curiosidade, aumentar a seriedade)
        if agent.epsilon > EPSILON_MIN:
            agent.epsilon *= EPSILON_DECAY

        # Relatório a cada 1000 jogos
        if episode % 1000 == 0:
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            elapsed = time.time() - start_time
            print(f"Episódio {episode:5d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Vitórias (últimos 1000): {win_rate:4.1f}% | "
                  f"Estados Explorados: {len(agent.q_table)}")

    # Fim do Treino
    total_time = time.time() - start_time
    print("-" * 50)
    print(f"✅ Treinamento concluído em {total_time:.1f} segundos!")
    
    # Passo 2.4: Persistência
    agent.save_model("brain.pkl")

if __name__ == "__main__":
    train()