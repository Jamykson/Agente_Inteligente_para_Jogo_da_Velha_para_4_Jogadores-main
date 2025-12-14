import streamlit as st
import numpy as np
import time
import random
from settings import *
from environment import TicTacToeEnv
from agent import QAgent

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Arena Neural 4x4",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CSS PERSONALIZADO ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica', sans-serif;
        text-align: center;
    }

    div.stButton > button {
        width: 100%;
        height: 80px;
        font-size: 30px;
        font-weight: bold;
        border-radius: 12px;
        border: 2px solid #262730;
        background-color: #1F2128;
        color: #FAFAFA;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    div.stButton > button:hover {
        border-color: #00C9FF;
        color: #00C9FF;
        transform: translateY(-2px);
    }

    div.stButton > button:disabled {
        background-color: #16181D;
        border-color: #16181D;
        opacity: 1.0;
        cursor: not-allowed;
    }
    
    div[data-testid="stMetricValue"] {
        text-align: center;
        color: #00C9FF;
    }
    div[data-testid="stMetricLabel"] {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- MAPA DE SÍMBOLOS (ATUALIZADO) ---
# 0: Vazio, 1: IA, 2: Humano, 3: Bot Monstro, 4: Bot Alien
SYMBOLS_MAP = {0: " ", 1: "🤖", 2: "😎", 3: "👾", 4: "👽"}

# --- LÓGICA DO JOGO ---

def manual_step(action, player_id):
    """Executa jogada lógica"""
    env = st.session_state.env
    row, col = divmod(action, BOARD_SIZE)
    env.board[row, col] = player_id
    
    if env.check_winner(player_id):
        return True, {'result': 'Win', 'winner': player_id}
    if env.is_draw():
        return True, {'result': 'Draw'}
    return False, {}

# --- INICIALIZAÇÃO ---
if 'env' not in st.session_state:
    st.session_state.env = TicTacToeEnv()
    st.session_state.board = st.session_state.env.board
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.status_text = "IA a calcular abertura..."
    st.session_state.turn_counter = 0
    
    st.session_state.agent = QAgent()
    try:
        st.session_state.agent.load_model("brain.pkl")
        st.session_state.agent.epsilon = 0.0 
    except:
        pass

# --- BARRA LATERAL ---
with st.sidebar:
    st.title("⚙️ Painel")
    st.markdown("---")
    # Legenda atualizada com os novos emojis
    st.write("🤖 **Agente IA** (Começa)")
    st.write("😎 **Você** (Humano)")
    st.write("👾 **Bot 3** (Monstro)")
    st.write("👽 **Bot 4** (Alien)")
    
    st.markdown("---")
    
    if st.button("🔄 Reiniciar Partida", type="primary"):
        st.session_state.env.reset()
        st.session_state.game_over = False
        st.session_state.status_text = "Novo Jogo!"
        st.session_state.turn_counter = 0
        st.rerun()

# --- ABERTURA DA IA ---
if st.session_state.turn_counter == 0 and not st.session_state.game_over:
    env = st.session_state.env
    agent = st.session_state.agent
    
    empty = [i for i in range(BOARD_SIZE**2) if env.is_valid_move(i)]
    
    # IA Joga Primeiro
    action = agent.choose_action(env.board, empty)
    manual_step(action, 1) # Jogador 1 (IA)
    
    st.session_state.turn_counter += 1
    st.session_state.status_text = "IA fez a abertura. Sua vez! (😎)"

# --- FUNÇÕES DE RENDERIZAÇÃO ---

def render_board(placeholder, interaction_enabled=True):
    """Desenha o grid 4x4"""
    board = st.session_state.env.board
    
    with placeholder.container():
        for r in range(BOARD_SIZE):
            cols = st.columns(BOARD_SIZE)
            for c in range(BOARD_SIZE):
                idx = r * BOARD_SIZE + c
                val = board[r, c]
                label = SYMBOLS_MAP[val]
                
                if interaction_enabled:
                    key_id = f"btn_interact_{idx}"
                else:
                    key_id = f"btn_anim_{idx}_{int(time.time()*100000)}"
                
                if interaction_enabled and val == 0 and not st.session_state.game_over:
                    if cols[c].button(" ", key=key_id):
                        run_turn_sequence(placeholder, idx)
                else:
                    cols[c].button(label, key=key_id, disabled=True)

def run_turn_sequence(placeholder, human_action):
    env = st.session_state.env
    st.session_state.turn_counter += 1
    
    # 1. JOGADA HUMANA
    done, info = manual_step(human_action, 2)
    st.session_state.status_text = "Você jogou... Aguarde."
    render_board(placeholder, interaction_enabled=False) 
    
    if check_end(done, info, "VOCÊ VENCEU!", "human"): return

    # 2. BOTs e IA (Sequência: Bot 3 -> Bot 4 -> IA)
    opponents = [3, 4, 1] 
    
    for player in opponents:
        time.sleep(1.5)
        
        name = SYMBOLS_MAP[player]
        st.session_state.status_text = f"Vez do {name}..."
        render_board(placeholder, interaction_enabled=False)
        
        empty = [i for i in range(BOARD_SIZE**2) if env.is_valid_move(i)]
        if not empty:
            check_end(True, {'result': 'Draw'}, "", "draw")
            return

        if player == 1:
            action = st.session_state.agent.choose_action(env.board, empty)
            win_msg = "A IA DOMINOU O JOGO!"
            winner_type = "ai"
        else:
            action = random.choice(empty)
            win_msg = f"BOT {name} GANHOU!"
            winner_type = "bot"
            
        done, info = manual_step(action, player)
        render_board(placeholder, interaction_enabled=False)
        
        if check_end(done, info, win_msg, winner_type): return
        
    st.session_state.status_text = "Sua Vez! (😎)"
    st.rerun()

def check_end(done, info, msg, winner_type):
    if done:
        st.session_state.game_over = True
        if info.get('result') == 'Win':
            st.session_state.status_text = msg
            if winner_type == "human":
                st.balloons()
            elif winner_type == "ai":
                st.snow()
        else:
            st.session_state.status_text = "EMPATE TÉCNICO!"
        st.rerun()
        return True
    return False

# --- INTERFACE PRINCIPAL ---

st.title("🧠 Velha Neural")

col1, col2 = st.columns(2)
col1.metric("Rodadas", st.session_state.turn_counter)
col2.metric("Status", st.session_state.status_text)

st.write("") 

board_placeholder = st.empty()
render_board(board_placeholder)