# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from modules.data_loader import DataLoader
from modules.ai_core import AICore
from modules.predictive_analysis import PredictiveAnalyzer
from modules.scenario_simulator import ScenarioSimulator
from modules.sentiment_analysis import SentimentAnalyzer
from modules.llm_interface import LocalLLM

st.set_page_config(page_title="Cogitara AI", page_icon="🚀", layout="wide")

# Simple CSS for look
st.markdown("""
<style>
    .main-header { font-size: 1.8rem; color: #7CFC00; text-align:center; margin-bottom:10px; }
    .stButton>button { border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# --- Helpers and singletons via session_state ---
if "data" not in st.session_state:
    st.session_state.data = None
if "ai" not in st.session_state:
    st.session_state.ai = None
if "llm" not in st.session_state:
    st.session_state.llm = None

data_loader = DataLoader()

# Sidebar: Upload + Init
with st.sidebar:
    st.title("🚀 Cogitara")
    uploaded = st.file_uploader("📁 Upload: CSV / XLSX / JSON", type=["csv","xlsx","json"])
    if uploaded is not None:
        try:
            st.session_state.data = data_loader.load_data(uploaded)
            st.success(f"✅ {len(st.session_state.data)} linhas carregadas")
            with st.expander("📋 Ver dados (preview)"):
                st.dataframe(st.session_state.data.head(200))
        except Exception as e:
            st.error(f"Erro ao carregar: {e}")

    st.markdown("---")
    init_btn = st.button("🔄 Inicializar Cogitara (local)")
    st.markdown("**LLM local** (opcional):")
    st.checkbox("Ativar LLM local (se houver pesos)", key="use_llm")

# Initialize AI stack
if init_btn:
    if st.session_state.data is None:
        st.warning("Envie um arquivo antes de inicializar.")
    else:
        with st.spinner("Inicializando componentes..."):
            ai_core = AICore()
            predictive = PredictiveAnalyzer()
            simulator = ScenarioSimulator()
            sentiment = SentimentAnalyzer()
            llm = None
            if st.session_state.use_llm:
                llm = LocalLLM()
                llm_status = llm.try_load_model()
                if not llm_status['ok']:
                    st.warning("LLM local não encontrado — usando fallback conversacional.")
            st.session_state.ai = {
                "core": ai_core,
                "predictive": predictive,
                "simulator": simulator,
                "sentiment": sentiment
            }
            st.session_state.llm = llm
            # teach core patterns
            ai_core.learn_data_patterns(st.session_state.data)
            st.success("✅ Cogitara inicializada")

# Top navigation
page = st.sidebar.radio("Navegação", ["Dashboard","Análise Preditiva","Simulador 'E se...'","Análise de Sentimento","Chat Conversacional","Análise Autônoma"])

# Dashboard
if page == "Dashboard":
    st.markdown('<div class="main-header">🏠 Dashboard — Cogitara</div>', unsafe_allow_html=True)
    if st.session_state.data is None:
        st.info("Faça upload dos dados no menu lateral para começar.")
    else:
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

# Predictive
elif page == "Análise Preditiva":
    st.markdown('<div class="main-header">📈 Análise Preditiva</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("Nenhuma coluna numérica encontrada.")
        else:
            target = st.selectbox("Variável alvo", num_cols, index=0)
            periods = st.number_input("Períodos de previsão (passos)", min_value=1, max_value=365, value=6)
            run = st.button("Executar previsão")
            if run:
                with st.spinner("Treinando e gerando previsão..."):
                    res = st.session_state.ai['predictive'].analyze(df, target, forecast_periods=int(periods), ai_core=st.session_state.ai['core'])
                    if 'error' in res:
                        st.error(res['error'])
                    else:
                        st.plotly_chart(res['forecast_plot'], use_container_width=True)
                        st.table(pd.DataFrame(res.get('forecast_table',{})))
                        for insight in res.get('insights',[]):
                            st.info(insight)

# Scenario simulator
elif page == "Simulador 'E se...'":
    st.markdown('<div class="main-header">🧭 Simulador de Cenários</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
    else:
        df = st.session_state.data
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            st.error("Nenhuma coluna numérica para simular.")
        else:
            st.write("Selecione variáveis e ajuste percentual (ex: +0.1 = +10%)")
            adjustments = {}
            cols_sel = st.multiselect("Variáveis", numeric, default=numeric[:2])
            for c in cols_sel:
                adjustments[c] = st.slider(f"Ajuste {c} (%)", -50, 100, 0, step=1)/100.0
            if st.button("Simular"):
                with st.spinner("Simulando..."):
                    sim = st.session_state.ai['simulator'].simulate(df, cols_sel, adjustments, ai_core=st.session_state.ai['core'])
                    if 'error' in sim:
                        st.error(sim['error'])
                    else:
                        st.write(f"Impacto médio: {sim['total_impact']:.2f}%")
                        st.plotly_chart(sim['comparison_plot'], use_container_width=True)
                        for r in sim.get('recommendations',[]):
                            st.success(r)

# Sentiment
elif page == "Análise de Sentimento":
    st.markdown('<div class="main-header">😊 Análise de Sentimento</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
    else:
        df = st.session_state.data
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if not text_cols:
            st.warning("Nenhuma coluna textual encontrada.")
        else:
            tcol = st.selectbox("Coluna de texto", text_cols)
            if st.button("Analisar Sentimento"):
                with st.spinner("Analisando..."):
                    out = st.session_state.ai['sentiment'].analyze(df, tcol, ai_core=st.session_state.ai['core'])
                    if 'error' in out:
                        st.error(out['error'])
                    else:
                        st.plotly_chart(out['sentiment_plot'], use_container_width=True)
                        for insight in out.get('insights',[]):
                            st.info(insight)

# Chat
elif page == "Chat Conversacional":
    st.markdown('<div class="main-header">💬 Chat — Cogitara</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
    else:
        llm = st.session_state.llm
        user_input = st.text_area("Pergunte algo sobre os dados (ex: 'Como estão as vendas?')", height=120)
        if st.button("Enviar"):
            with st.spinner("Cogitara pensando..."):
                # If LLM available, use it
                if llm and llm.is_ready():
                    answer = llm.generate_response(user_input, context_data=st.session_state.data, ai_core=st.session_state.ai['core'])
                    st.markdown("**Cogitara:**")
                    st.write(answer)
                else:
                    # fallback: basic reasoning using modules
                    # try keyword mapping
                    lower = user_input.lower()
                    if "venda" in lower or "fatur" in lower:
                        # simple summary
                        num_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
                        if num_cols:
                            col = num_cols[0]
                            avg = st.session_state.data[col].mean()
                            st.write(f"Resumo rápido: a média de `{col}` é {avg:.2f}. Use Análise Preditiva para projeções detalhadas.")
                        else:
                            st.write("Nenhuma métrica numérica disponível.")
                    elif "sentiment" in lower or "sentimento" in lower or "coment" in lower:
                        text_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
                        if text_cols:
                            out = st.session_state.ai['sentiment'].quick_analyze(st.session_state.data, text_cols[0])
                            st.write(f"Estimativa (fallback): Pos {out.get('positive',0):.1f}% | Neu {out.get('neutral',0):.1f}% | Neg {out.get('negative',0):.1f}%")
                        else:
                            st.write("Nenhuma coluna textual para analisar.")
                    else:
                        st.write("Desculpe — sem LLM local ativo. Perguntas gerais sobre dados: tente palavras-chave como 'vendas', 'sentimento', 'previsão'.")
# Autonomous analysis
elif page == "Análise Autônoma":
    st.markdown('<div class="main-header">🤖 Análise Autônoma</div>', unsafe_allow_html=True)
    if not st.session_state.ai:
        st.warning("Inicialize Cogitara primeiro.")
    else:
        if st.button("Executar Análise Completa"):
            with st.spinner("Executando pipeline autônomo..."):
                ai_obj = st.session_state.ai
                core = ai_obj['core']
                summary = core.quick_summary(st.session_state.data)
                recommendations = core.generate_recommendations(st.session_state.data)
                alerts = core.detect_alerts(st.session_state.data)
                st.subheader("Resumo Executivo")
                st.table(pd.DataFrame([summary]))
                st.subheader("Recomendações")
                for rec in recommendations:
                    st.success(rec)
                if alerts:
                    st.subheader("Alertas")
                    for a in alerts:
                        st.warning(a)

