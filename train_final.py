# train_final.py
import numpy as np
import time
import os
from environment import TicTacToeEnv
from agent import QAgent
from settings import *

def train_grandmaster():
    print("👑 INICIANDO O TREINO SUPREMO (BATTLE ROYALE) 👑")
    print("Cenário: [2: Elite] | [3: Veterano] | [4: Louco]")
    
    # --- 1. PREPARAR A MESA DE OPONENTES ---
    brains_map = {}
    
    # Oponente 2: Tenta carregar o Elite (Fase 2)
    if os.path.exists("brain_v2_elite.pkl"):
        print("💀 [Assento 2] Mestre Elite: CARREGADO")
        bot_elite = QAgent()
        bot_elite.load_model("brain_v2_elite.pkl")
        brains_map[2] = bot_elite
    else:
        print("⚠️ [Assento 2] Elite não encontrado -> Usando Aleatório.")

    # Oponente 3: Tenta carregar o Veterano (Fase 1)
    if os.path.exists("brain.pkl"):
        print("🤖 [Assento 3] Veterano: CARREGADO")
        bot_veteran = QAgent()
        bot_veteran.load_model("brain.pkl")
        brains_map[3] = bot_veteran
    else:
        print("⚠️ [Assento 3] Veterano não encontrado -> Usando Aleatório.")
        
    # Oponente 4: Sempre Aleatório (O fator Caos)
    brains_map[4] = None 

    # --- 2. PREPARAR O NOSSO CAMPEÃO ---
    champion = QAgent()
    
    # Ele deve continuar evoluindo do ponto mais forte que você tiver
    if os.path.exists("brain_v2_elite.pkl"):
        print("🛡️ Campeão: Continuando do nível Elite...")
        champion.load_model("brain_v2_elite.pkl")
    elif os.path.exists("brain.pkl"):
        print("🛡️ Campeão: Continuando do nível Veterano...")
        champion.load_model("brain.pkl")
    else:
        print("🐣 Campeão: Começando do zero (Vai demorar mais).")
    
    # Ajuste Fino: Curiosidade baixa, aprendizado cirúrgico
    champion.epsilon = 0.2  
    champion.alpha = 0.05   

    # INICIALIZA O AMBIENTE COM A MESA MISTA
    env = TicTacToeEnv(opponent_brains=brains_map)
    
    print("-" * 50)
    start_time = time.time()
    recent_wins = [] 
    
    # Loop de Treino
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        state_matrix = env.board.copy()
        done = False
        
        while not done:
            valid_moves = [i for i in range(BOARD_SIZE * BOARD_SIZE) if env.is_valid_move(i)]
            if not valid_moves: break

            action = champion.choose_action(state_matrix, valid_moves)
            
            # Environment escolhe quem joga contra (Elite, Veterano ou Random)
            next_state_flat, reward, done, info = env.step(action)
            next_state_matrix = env.board.copy()

            champion.learn(state_matrix, action, reward, next_state_matrix)
            state_matrix = next_state_matrix

        # Estatísticas
        if info.get('result') == 'Win':
            recent_wins.append(1)
        else:
            recent_wins.append(0) # Derrota ou Empate = 0
            
        if len(recent_wins) > 1000: recent_wins.pop(0)

        champion.decay_alpha()
        if champion.epsilon > 0.001:
            champion.epsilon *= 0.99995

        if episode % 1000 == 0:
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            print(f"Episódio {episode:6d} | Win Rate: {win_rate:5.1f}% | Eps: {champion.epsilon:.3f} | Q-Table: {len(champion.q_table)}")

    total_time = time.time() - start_time
    print("-" * 50)
    print(f"✅ TREINO SUPREMO CONCLUÍDO ({total_time:.1f}s)")
    
    # Salva o arquivo FINAL DEFINITIVO
    champion.save_model("brain_final_boss.pkl")
    print("💾 CÉREBRO FINAL SALVO: 'brain_final_boss.pkl'")

if __name__ == "__main__":
    train_grandmaster()