import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="In silico модель CHO", layout="wide")

st.title("🔬 Динамическая In silico модель процесса биосинтеза леканемаба")
st.markdown(" Цифровой двойник Fed-Batch культивирования клеток CHO")

# ==============================================================================
# БОКОВАЯ ПАНЕЛЬ — НАСТРОЙКИ ПОДПИТКИ И КИНЕТИКИ (КАЛИБРОВАННЫЕ)
# ==============================================================================
st.sidebar.header("⚙️ Параметры подпитки (Fed-Batch)")
G_FEED = st.sidebar.number_input("🍼 Глюкоза в подпитке (G_feed), [г/л]", value=120.0, step=5.0)
GLN_FEED = st.sidebar.number_input("🧪 Глутамин в подпитке (Gln_feed), [г/л]", value=35.0, step=1.0)
T_FEED_START = st.sidebar.number_input("⏳ Старт подачи насоса, [ч]", value=36, step=6)
FEED_RATE = st.sidebar.number_input("💧 Скорость насоса (F), [л/ч]", value=0.0180, step=0.001, format="%.4f")

st.sidebar.header("🧫 Базовая кинетика штамма")
MU_MAX = st.sidebar.number_input("📈 Макс. скорость роста (MU_MAX), [1/ч]", value=0.0400, format="%.4f")
KD     = st.sidebar.number_input("💀 Скорость гибели (KD), [1/ч]", value=0.0020, format="%.4f") 
Q_P    = st.sidebar.number_input("💎 Экспрессия антител (Q_P), [г/(г*ч)]", value=0.0300, format="%.4f")

st.sidebar.subheader("💨 Массообмен по кислороду")
KLA    = st.sidebar.number_input("🔄 Коэф. массопередачи (kLa), [1/ч]", value=15.0, step=0.5, format="%.1f")
Q_O2   = st.sidebar.number_input("🫁 Скорость дыхания (q_O2), [% / (г*ч)]", value=2.5, step=0.1, format="%.2f")

st.sidebar.header("🧪 Стартовая среда в реакторе")
G0     = st.sidebar.number_input("🥛 Стартовая глюкоза (G0), [г/л]", value=5.5, format="%.1f")
Gln0   = st.sidebar.number_input("🧪 Стартовый глутамин (Gln0), [г/л]", value=1.2, format="%.1f")
V0     = st.sidebar.number_input("📐 Начальный объем (V0), [л]", value=7.5, format="%.1f")
T_END  = st.sidebar.number_input("⏳ Время процесса, [часы]", value=144, step=12)

# ==============================================================================
# МАТЕМАТИКА СДУ (FED-BATCH + O2)
# ==============================================================================
Y_XG, Y_XGLN, Y_LG, Y_AGLN = 0.4, 0.2, 0.8, 0.15
M_G, M_GLN = 0.002, 0.001
K_G, K_GLN, K_O2, KI_L, KI_A = 0.35, 0.08, 5.0, 3.5, 0.08
O2_SAT = 100.0

# 8 переменных: [Xv, G, Gln, L, A, P, O2, V]
INIT_CONDITIONS = [0.2, G0, Gln0, 0.0, 0.0, 0.0, 100.0, V0]

def fed_batch_ode(t, y):
    Xv, G, Gln, L, A, P, O2, V = y
    
    G_eff, Gln_eff, O2_eff = max(0.0, G), max(0.0, Gln), max(0.0, O2)
    
    # Логика работы дозирующего насоса
    if t >= T_FEED_START and V < 10.0:
        F = FEED_RATE
    else:
        F = 0.0
        
    # Уравнение Моно со всеми 5 факторами (включая КИСЛОРОД)
    mu = MU_MAX * (G_eff/(K_G + G_eff)) * (Gln_eff/(K_GLN + Gln_eff)) * (O2_eff/(K_O2 + O2_eff)) * (KI_L/(KI_L + L)) * (KI_A/(KI_A + A))
    
    # Дифференциальные уравнения
    dXv_dt  = (mu - KD) * Xv - (F / V) * Xv
    dG_dt   = - ((mu / Y_XG) + M_G) * Xv + (F / V) * (G_FEED - G)
    dGln_dt = - ((mu / Y_XGLN) + M_GLN) * Xv + (F / V) * (GLN_FEED - Gln)
    dL_dt   = Y_LG * ((mu / Y_XG) * Xv) - (F / V) * L
    dA_dt   = Y_AGLN * ((mu / Y_XGLN) * Xv) - (F / V) * A
    dP_dt   = Q_P * Xv - (F / V) * P
    dO2_dt  = KLA * (O2_SAT - O2) - Q_O2 * Xv  # <--- КИСЛОРОД ТУТ!
    dV_dt   = F
    
    return [dXv_dt, dG_dt, dGln_dt, dL_dt, dA_dt, dP_dt, dO2_dt, dV_dt]

t_eval = np.linspace(0, T_END, 1000)
solution = solve_ivp(fed_batch_ode, (0, T_END), INIT_CONDITIONS, t_eval=t_eval, method='RK45')
Xv_res, G_res, Gln_res, L_res, A_res, P_res, O2_res, V_res = solution.y

# ==============================================================================
# ИНТЕРФЕЙС И ВЫВОД РЕЗУЛЬТАТОВ
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Показатели Fed-Batch серии")
    st.metric("Финальный титр Леканемаба", f"{P_res[-1]:.3f} г/л")
    st.metric("Финальный объем культуры", f"{V_res[-1]:.2f} л")
    st.metric("Максимальная плотность клеток", f"{max(Xv_res):.2f} г/л")
    st.metric("Минимальный уровень pO₂", f"{min(O2_res):.1f} %")
    
    total_grams = P_res[-1] * V_res[-1]
    st.success(f"Реальный выход субстанции с серии ({V_res[-1]:.1f} л): **{total_grams:.2f} г** чистого белка")

with col2:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    
    ax1.plot(solution.t, Xv_res, 'g-', label='Живая биомасса Xv (г/л)', lw=2)
    ax1.plot(solution.t, P_res, 'b-', label='Леканемаб P (г/л)', lw=2)
    ax1.grid(True, alpha=0.3); ax1.legend()
    ax1.set_title("Динамика Цифрового Двойника процесса")
    
    ax2.plot(solution.t, G_res, 'c--', label='Глюкоза G (г/л)')
    ax2.plot(solution.t, Gln_res, 'm--', label='Глутамин Gln (г/л)')
    ax2.grid(True, alpha=0.3); ax2.legend()
    
    ax3.plot(solution.t, O2_res, 'orange', label='Растворенный кислород pO₂ (%)', lw=2)
    ax3.axhline(30.0, color='red', linestyle=':', label='Критический порог (30%)')
    ax3.grid(True, alpha=0.3); ax3.legend()
    
    ax4.plot(solution.t, V_res, 'purple', label='Объем культуры V (л)', lw=2)
    ax4.set_xlabel('Время процесса, часы')
    ax4.grid(True, alpha=0.3); ax4.legend()
    
    st.pyplot(fig)
