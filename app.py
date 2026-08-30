import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Настройка страницы в браузере
st.set_page_config(page_title="In silico модель биореактора CHO", layout="wide")

st.title(" Компьютерное моделирование (In silico) кинетики культивирования клеток CHO")
st.markdown("Динамический цифровой двойник процесса биосинтеза биосимиляра леканемаба с учетом массопередачи O₂")

# ==============================================================================
# БОКОВАЯ ПАНЕЛЬ (SIDEBAR) — ВВОД ДАННЫХ ЦИФРАМИ С ЕДИНИЦАМИ ИЗМЕРЕНИЯ
# ==============================================================================
st.sidebar.header(" Кинетические константы")

MU_MAX = st.sidebar.number_input(" Макс. скорость роста (MU_MAX), [1/ч]", min_value=0.001, max_value=0.2, value=0.0350, step=0.0005, format="%.4f")
KD     = st.sidebar.number_input(" Скорость гибели (KD), [1/ч]", min_value=0.0001, max_value=0.05, value=0.0040, step=0.0001, format="%.4f")

st.sidebar.subheader(" Лимитирование (Константы Моно)")
K_G    = st.sidebar.number_input(" По глюкозе (K_G), [г/л]", min_value=0.01, max_value=5.0, value=0.3500, step=0.01, format="%.4f")
K_GLN  = st.sidebar.number_input(" По глутамину (K_GLN), [г/л]", min_value=0.01, max_value=2.0, value=0.0800, step=0.005, format="%.4f")
# НОВАЯ КОНСТАНТА: Константа Моно по кислороду (критический порог, ниже которого клетка задыхается)
K_O2   = st.sidebar.number_input(" По кислороду (K_O2), [% от насыщения]", min_value=1.0, max_value=20.0, value=5.0, step=0.5, format="%.1f")

st.sidebar.subheader(" Ингибирование метаболитами")
KI_L   = st.sidebar.number_input(" Лактатом (KI_L), [г/л]", min_value=0.5, max_value=20.0, value=3.5000, step=0.1, format="%.4f")
KI_A   = st.sidebar.number_input(" Аммонием (KI_A), [г/л]", min_value=0.01, max_value=1.0, value=0.0800, step=0.005, format="%.4f")

st.sidebar.subheader(" Продуктивность и массообмен")
Q_P    = st.sidebar.number_input(" Экспрессия антител (Q_P), [г/(г клеток * ч)]", min_value=0.0001, max_value=0.01, value=0.0012, step=0.0001, format="%.4f")
# Инженерные параметры аэрации волнового мешка
KLA    = st.sidebar.number_input(" Коэф. массопередачи кислорода (kLa), [1/ч]", min_value=1.0, max_value=50.0, value=15.0, step=0.5, format="%.1f")
Q_O2   = st.sidebar.number_input(" Скорость дыхания клеток (q_O2), [% / (г клеток * ч)]", min_value=0.5, max_value=10.0, value=2.5, step=0.1, format="%.2f")

st.sidebar.header("Стартовые параметры среды")
G0     = st.sidebar.number_input(" Начальная глюкоза (G0), [г/л]", min_value=1.0, max_value=25.0, value=6.0000, step=0.1, format="%.4f")
Gln0   = st.sidebar.number_input(" Начальный глутамин (Gln0), [г/л]", min_value=0.1, max_value=5.0, value=1.2000, step=0.05, format="%.4f")
O2_0   = st.sidebar.number_input(" Начальный кислород (pО2), [%]", min_value=10.0, max_value=100.0, value=100.0, step=5.0, format="%.1f")
T_END  = st.sidebar.number_input(" Время процесса, [часы]", min_value=12, max_value=360, value=144, step=12)

# ==============================================================================
# МАТЕМАТИЧЕСКИЙ РАСЧЕТ СДУ С КИСЛОРОДОМ
# ==============================================================================
# Постоянные выходы биомассы и поддержания
Y_XG, Y_XGLN, Y_LG, Y_AGLN = 0.4, 0.2, 0.8, 0.15
M_G, M_GLN = 0.002, 0.001
O2_SAT = 100.0  # Уровень насыщения кислородом в газовой подушке мешка (%)

# Вектор состояния: [Xv, G, Gln, L, A, P, O2]
INIT_CONDITIONS = [0.2, G0, Gln0, 0.0, 0.0, 0.0, O2_0]

def cho_bioreactor_ode(t, y):
    Xv, G, Gln, L, A, P, O2 = y
    
    # Защитные отсечки для численных методов
    G_eff   = max(0.0, G)
    Gln_eff = max(0.0, Gln)
    O2_eff  = max(0.0, O2)
    
    # Расчет скорости роста Моно с учетом НОВОГО члена по кислороду (O2_eff / (K_O2 + O2_eff))
    mu = MU_MAX * (G_eff/(K_G + G_eff)) * (Gln_eff/(K_GLN + Gln_eff)) * (O2_eff/(K_O2 + O2_eff)) * (KI_L/(KI_L + L)) * (KI_A/(KI_A + A))
    
    # Система дифференциальных уравнений
    dXv_dt  = (mu - KD) * Xv
    dG_dt   = - ((mu / Y_XG) + M_G) * Xv
    dGln_dt = - ((mu / Y_XGLN) + M_GLN) * Xv
    dL_dt   = Y_LG * ((mu / Y_XG) * Xv)
    dA_dt   = Y_AGLN * ((mu / Y_XGLN) * Xv)
    dP_dt   = Q_P * Xv
    
    # УРАВНЕНИЕ КИСЛОРОДА: Изменение pO2 = (Массопередача из газа в жидкость) - (Потребление клетками на дыхание)
    dO2_dt  = KLA * (O2_SAT - O2) - Q_O2 * Xv
    
    return [dXv_dt, dG_dt, dGln_dt, dL_dt, dA_dt, dP_dt, dO2_dt]

# Интегрирование системы
t_eval = np.linspace(0, T_END, 1000)
solution = solve_ivp(cho_bioreactor_ode, (0, T_END), INIT_CONDITIONS, t_eval=t_eval, method='RK45')
Xv_res, G_res, Gln_res, L_res, A_res, P_res, O2_res = solution.y

# ==============================================================================
# ИНТЕРФЕЙС И ВЫВОД РЕЗУЛЬТАТОВ
# ==============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Технологические показатели серии")
    st.metric("Финальный титр Леканемаба", f"{P_res[-1]:.3f} г/л")
    st.metric("Максимальная плотность клеток", f"{max(Xv_res):.2f} г/л")
    st.metric("Остаточная глюкоза", f"{G_res[-1]:.3f} г/л")
    st.metric("Финальный уровень pO₂", f"{O2_res[-1]:.1f} %")
    
    total_grams = P_res[-1] * 10
    st.success(f"Выход активной субстанции с 10-литровой серии: **{total_grams:.2f} г** чистейшего белка")

with col2:
    # Строим три графика, чтобы кислород был виден отдельно и красиво
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # График 1: Биомасса и целевой белок
    ax1.plot(solution.t, Xv_res, 'g-', label='Живая биомасса Xv (г/л)', lw=2.5)
    ax1.plot(solution.t, P_res, 'b-', label='Леканемаб P (г/л)', lw=2.5)
    ax1.set_ylabel('Концентрация, г/л')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left')
    ax1.set_title('Кинетика биосинтеза в волновом биореакторе')
    
    # График 2: Питание и токсины
    ax2.plot(solution.t, G_res, 'c--', label='Глюкоза G (г/л)', lw=1.8)
    ax2.plot(solution.t, Gln_res, 'm--', label='Глутамин Gln (г/л)', lw=1.8)
    ax2.plot(solution.t, L_res, 'r-.', label='Лактат L (г/л)', lw=1.8)
    ax2.set_ylabel('Концентрация, г/л')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # График 3: Растворенный кислород
    ax3.plot(solution.t, O2_res, 'orange', label='Растворенный кислород pO₂ (%)', lw=2.5)
    ax3.axhline(30.0, color='red', linestyle=':', label='Критический порог pO₂ (30%)')
    ax3.set_xlabel('Время процесса, часы')
    ax3.set_ylabel('Уровень pO₂, %')
    ax3.set_ylim(-5, 105)
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend(loc='lower left')
    
    st.pyplot(fig)

