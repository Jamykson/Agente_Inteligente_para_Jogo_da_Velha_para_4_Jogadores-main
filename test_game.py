import random
import time
from environment import TicTacToeEnv
from settings import BOARD_SIZE

def run_random_games(num_games=100):
    env = TicTacToeEnv()
    stats = {'Win': 0, 'Loss': 0, 'Draw': 0, 'Invalid': 0}

    print(f"--- Iniciando Teste de Fumaça: {num_games} Partidas Aleatórias ---")
    start_time = time.time()

    for i in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            # Agente "Burro": Escolhe qualquer número de 0 a 15
            # Isso vai testar se a lógica de 'Jogada Inválida' funciona também
            action = random.randint(0, (BOARD_SIZE * BOARD_SIZE) - 1)
            
            next_state, reward, done, info = env.step(action)
            
            if done:
                if 'result' in info:
                    stats[info['result']] += 1
                elif reward == -50: # Se o jogo acabar por muitas jogadas invalidas (opcional)
                     stats['Invalid'] += 1

    end_time = time.time()
    
    print("\n--- Resultados do Teste ---")
    print(f"Tempo total: {end_time - start_time:.4f} segundos")
    print(f"Vitórias do Agente (Sorte): {stats.get('Win', 0)}")
    print(f"Derrotas (Oponentes ganharam): {stats.get('Loss', 0)}")
    print(f"Empates: {stats.get('Draw', 0)}")
    
    if stats['Win'] + stats['Loss'] + stats['Draw'] == num_games:
        print("\n✅ SUCESSO: Todas as partidas terminaram corretamente.")
        print("A Fase 1 está concluída. Pode prosseguir para a IA.")
    else:
        print("\n❌ ERRO: Algumas partidas não tiveram desfecho claro.")

if __name__ == "__main__":
    run_random_games()