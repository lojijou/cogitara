# app.py (versão refatorada)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from modules.data_loader import DataLoader
from modules.ai_core import AICore
from modules.predictive_analysis import PredictiveAnalyzer
from modules.scenario_simulator import ScenarioSimulator
from modules.sentiment_analysis import SentimentAnalyzer
# LLM opcional
try:
    from modules.llm_interface import LocalLLM
except ImportError:
    LocalLLM = None

# --- Configuração da página ---
st.set_page_config(page_title="Cogitara AI", page_icon="🚀", layout="wide")
st.markdown("""
<style>
    .main-header { font-size:1.8rem; color:#7CFC00; text-align:center; margin-bottom:10px;}
    .stButton>button { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "data" not in st.session_state: st.session_state.data = None
if "ai" not in st.session_state: st.session_state.ai = None
if "llm" not in st.session_state: st.session_state.llm = None

data_loader = DataLoader()

# --- Sidebar: Upload + Inicialização ---
with st.sidebar:
    st.title("🚀 Cogitara")
    uploaded = st.file_uploader("📁 Upload: CSV / XLSX / JSON", type=["csv","xlsx","json"])
    
    if uploaded:
        try:
            st.session_state.data = data_loader.load_data(uploaded)
            st.success(f"✅ {len(st.session_state.data)} linhas carregadas")
            with st.expander("📋 Ver dados (preview)"):
                st.dataframe(st.session_state.data.head(200))
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")

    st.markdown("---")
    init_form = st.form("init_form")
    init_btn = init_form.form_submit_button("🔄 Inicializar Cogitara (local)")
    use_llm = init_form.checkbox("Ativar LLM local (se houver pesos)")
    init_form.form_submit_button("Confirmar")

# --- Inicialização do stack de IA ---
if init_btn:
    if st.session_state.data is None:
        st.warning("Envie um arquivo antes de inicializar.")
    else:
        with st.spinner("Inicializando componentes..."):
            ai_core = AICore()
            predictive = PredictiveAnalyzer()
            simulator = ScenarioSimulator()
            sentiment = SentimentAnalyzer()
            llm = LocalLLM() if LocalLLM and use_llm else None
            if llm:
                llm_status = llm.try_load_model()
                if not llm_status.get('ok', False):
                    st.warning("LLM local não encontrado — fallback conversacional.")
            st.session_state.ai = {
                "core": ai_core,
                "predictive": predictive,
                "simulator": simulator,
                "sentiment": sentiment
            }
            st.session_state.llm = llm
            ai_core.learn_data_patterns(st.session_state.data)
            st.success("✅ Cogitara inicializada")

# --- Navegação ---
page = st.sidebar.radio("Navegação", [
    "Dashboard","Análise Preditiva","Simulador 'E se...'","Análise de Sentimento",
    "Chat Conversacional","Análise Autônoma"
])

# --- Páginas ---
def dashboard_page():
    st.markdown('<div class="main-header">🏠 Dashboard — Cogitara</div>', unsafe_allow_html=True)
    if st.session_state.data is None:
        st.info("Faça upload dos dados no menu lateral para começar.")
        return
    df = st.session_state.data
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Linhas", f"{len(df):,}")
    col2.metric("Colunas", len(df.columns))
    col3.metric("Numéricas", len(df.select_dtypes(include=[np.number]).columns))
    col4.metric("Textuais", len(df.select_dtypes(include=['object']).columns))
    st.markdown("### Visualização rápida")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        c = st.selectbox("Coluna numérica", num_cols, key="dashboard_col")
        fig = px.line(df.reset_index(), y=c, x=df.index, title=f"Evolução de {c}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhuma coluna numérica para gráfico.")

def predictive_page():
    st.markdown('<div class="main-header">📈 Análise Preditiva</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
        return
    df = st.session_state.data
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("Nenhuma coluna numérica encontrada.")
        return
    target = st.selectbox("Variável alvo", num_cols)
    periods = st.number_input("Períodos de previsão", min_value=1, max_value=365, value=6)
    with st.form("predictive_form"):
        run = st.form_submit_button("Executar previsão")
    if run:
        with st.spinner("Treinando e gerando previsão..."):
            res = st.session_state.ai['predictive'].analyze(df, target, forecast_periods=int(periods),
                                                           ai_core=st.session_state.ai['core'])
            if 'error' in res: st.error(res['error'])
            else:
                st.plotly_chart(res['forecast_plot'], use_container_width=True)
                st.table(pd.DataFrame(res.get('forecast_table',{})))
                for insight in res.get('insights',[]): st.info(insight)

def simulator_page():
    st.markdown('<div class="main-header">🧭 Simulador de Cenários</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
        return
    df = st.session_state.data
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        st.error("Nenhuma coluna numérica para simular.")
        return
    st.write("Selecione variáveis e ajuste percentual (ex: +0.1 = +10%)")
    adjustments = {}
    cols_sel = st.multiselect("Variáveis", numeric, default=numeric[:2])
    for c in cols_sel:
        adjustments[c] = st.slider(f"Ajuste {c} (%)", -50, 100, 0, step=1)/100.0
    with st.form("sim_form"):
        run = st.form_submit_button("Simular")
    if run:
        with st.spinner("Simulando..."):
            sim = st.session_state.ai['simulator'].simulate(df, cols_sel, adjustments,
                                                            ai_core=st.session_state.ai['core'])
            if 'error' in sim: st.error(sim['error'])
            else:
                st.write(f"Impacto médio: {sim['total_impact']:.2f}%")
                st.plotly_chart(sim['comparison_plot'], use_container_width=True)
                for r in sim.get('recommendations',[]): st.success(r)

def sentiment_page():
    st.markdown('<div class="main-header">😊 Análise de Sentimento</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
        return
    df = st.session_state.data
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not text_cols:
        st.warning("Nenhuma coluna textual encontrada.")
        return
    tcol = st.selectbox("Coluna de texto", text_cols)
    with st.form("sentiment_form"):
        run = st.form_submit_button("Analisar Sentimento")
    if run:
        with st.spinner("Analisando..."):
            out = st.session_state.ai['sentiment'].analyze(df, tcol, ai_core=st.session_state.ai['core'])
            if 'error' in out: st.error(out['error'])
            else:
                st.plotly_chart(out['sentiment_plot'], use_container_width=True)
                for insight in out.get('insights',[]): st.info(insight)

def chat_page():
    st.markdown('<div class="main-header">💬 Chat — Cogitara</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
        return
    llm = st.session_state.llm
    user_input = st.text_area("Pergunte algo sobre os dados", height=120)
    with st.form("chat_form"):
        send = st.form_submit_button("Enviar")
    if send:
        with st.spinner("Cogitara pensando..."):
            if llm and llm.is_ready():
                answer = llm.generate_response(user_input, context_data=st.session_state.data,
                                               ai_core=st.session_state.ai['core'])
                st.markdown("**Cogitara:**")
                st.write(answer)
            else:
                st.write("Fallback: análise básica sem LLM.")
                st.write("Tente palavras-chave como 'vendas', 'sentimento', 'previsão'.")

def autonomous_page():
    st.markdown('<div class="main-header">🤖 Análise Autônoma</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
        return
    with st.form("auto_form"):
        run = st.form_submit_button("Executar Análise Completa")
    if run:
        with st.spinner("Executando pipeline autônomo..."):
            ai_obj = st.session_state.ai
            core = ai_obj['core']
            summary = core.quick_summary(st.session_state.data)
            recommendations = core.generate_recommendations(st.session_state.data)
            alerts = core.detect_alerts(st.session_state.data)
            st.subheader("Resumo Executivo")
            st.table(pd.DataFrame([summary]))
            st.subheader("Recomendações")
            for rec in recommendations: st.success(rec)
            if alerts:
                st.subheader("Alertas")
                for a in alerts: st.warning(a)

# --- Mapear páginas ---
pages = {
    "Dashboard": dashboard_page,
    "Análise Preditiva": predictive_page,
    "Simulador 'E se...'": simulator_page,
    "Análise de Sentimento": sentiment_page,
    "Chat Conversacional": chat_page,
    "Análise Autônoma": autonomous_page
}

pages.get(page, lambda: st.write("Página não implementada"))()
