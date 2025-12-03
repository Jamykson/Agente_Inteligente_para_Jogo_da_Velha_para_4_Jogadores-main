# train.py
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
    recent_wins = [] 
    
    for episode in range(1, EPISODES + 1):
        # MODO DE TESTE FINAL: Desliga aleatoriedade nos últimos 5% dos jogos
        if episode > EPISODES * 0.95:
            agent.epsilon = 0.0

        state = env.reset()
        state_matrix = env.board
        done = False
        
        while not done:
            valid_moves = [i for i in range(BOARD_SIZE * BOARD_SIZE) if env.is_valid_move(i)]
            if not valid_moves: break

            action = agent.choose_action(state_matrix, valid_moves)
            next_state_flat, reward, done, info = env.step(action)
            next_state_matrix = env.board

            agent.learn(state_matrix, action, reward, next_state_matrix)
            state_matrix = next_state_matrix.copy()

        # Estatísticas
        if info.get('result') == 'Win':
            recent_wins.append(1)
        else:
            recent_wins.append(0)
            
        if len(recent_wins) > 1000:
            recent_wins.pop(0)

        # Decaimento padrão (só acontece se não estivermos no modo de teste final)
        if episode <= EPISODES * 0.95:
            if agent.epsilon > EPSILON_MIN:
                agent.epsilon *= EPSILON_DECAY
        
        agent.decay_alpha()

        if episode % 1000 == 0:
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            
            # Marcador visual para saber se está no modo "Validação Pura"
            mode = "TESTE" if agent.epsilon == 0.0 else "TREINO"
            
            print(f"Episódio {episode:6d} [{mode}] | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Alpha: {agent.alpha:.3f} | "
                  f"Vitórias: {win_rate:4.1f}% | "
                  f"Estados: {len(agent.q_table)}")

    total_time = time.time() - start_time
    print("-" * 50)
    print(f"✅ Treinamento concluído em {total_time:.1f} segundos!")
    agent.save_model("brain.pkl")

if __name__ == "__main__":
    train()