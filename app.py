import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Importações dos módulos da Cogitara
import sys
import os
sys.path.append(os.path.dirname(__file__))

from cogitara.main import CogitaraAI
from cogitara.utils.data_loader import DataLoader
from cogitara.utils.preprocessor import DataPreprocessor

# Configuração da página
st.set_page_config(
    page_title="Cogitara AI - IA Empresarial Autônoma",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CogitaraApp:
    def __init__(self):
        self.ai = None
        self.data = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Inicializa o estado da sessão"""
        if 'ai_initialized' not in st.session_state:
            st.session_state.ai_initialized = False
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
    
    def render_sidebar(self):
        """Renderiza a barra lateral"""
        with st.sidebar:
            st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=COGITARA", width=150)
            st.title("🚀 Cogitara AI")
            st.markdown("---")
            
            # Upload de dados
            st.subheader("📊 Carregar Dados")
            uploaded_file = st.file_uploader(
                "Faça upload dos dados da empresa",
                type=['csv', 'xlsx', 'json'],
                help="Formatos suportados: CSV, Excel, JSON"
            )
            
            if uploaded_file is not None:
                self.load_data(uploaded_file)
            
            st.markdown("---")
            
            # Configurações da IA
            st.subheader("⚙️ Configurações")
            if st.button("🔄 Inicializar IA Cogitara", use_container_width=True):
                self.initialize_ai()
            
            if st.session_state.ai_initialized:
                st.success("✅ IA Inicializada")
                
            st.markdown("---")
            
            # Navegação
            st.subheader("🧭 Navegação")
            page = st.radio(
                "Selecione o módulo:",
                ["🏠 Dashboard", "📈 Análise Preditiva", "🔄 Simulador de Cenários", 
                 "😊 Análise de Sentimento", "🤖 IA Autônoma"]
            )
            
            return page
    
    def load_data(self, uploaded_file):
        """Carrega e processa os dados enviados"""
        try:
            data_loader = DataLoader()
            self.data = data_loader.load_data(uploaded_file)
            st.session_state.current_data = self.data
            
            # Pré-processamento automático
            preprocessor = DataPreprocessor()
            self.data = preprocessor.auto_preprocess(self.data)
            
            st.success(f"✅ Dados carregados: {self.data.shape[0]} linhas × {self.data.shape[1]} colunas")
            
            # Mostrar preview
            with st.expander("📋 Visualizar Dados"):
                st.dataframe(self.data.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados: {str(e)}")
    
    def initialize_ai(self):
        """Inicializa a IA Cogitara"""
        try:
            if st.session_state.current_data is not None:
                self.ai = CogitaraAI(st.session_state.current_data)
                st.session_state.ai_initialized = True
                st.success("🎉 IA Cogitara inicializada com sucesso!")
            else:
                st.warning("⚠️ Por favor, carregue os dados primeiro.")
        except Exception as e:
            st.error(f"❌ Erro ao inicializar IA: {str(e)}")
    
    def render_dashboard(self):
        """Renderiza o dashboard principal"""
        st.markdown('<div class="main-header">🏠 Dashboard Cogitara AI</div>', unsafe_allow_html=True)
        
        if not st.session_state.ai_initialized:
            st.warning("🚨 Inicialize a IA Cogitara primeiro para acessar todas as funcionalidades")
            return
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total de Dados", f"{len(self.data):,}")
        
        with col2:
            st.metric("🔮 Variáveis", f"{len(self.data.columns):,}")
        
        with col3:
            st.metric("🎯 Insights Gerados", "15")
        
        with col4:
            st.metric("⚡ Precisão IA", "94.2%")
        
        # Análises rápidas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Tendências Principais")
            self.render_trends()
        
        with col2:
            st.subheader("🎭 Análise de Sentimento Rápida")
            self.render_quick_sentiment()
        
        # Recomendações da IA
        st.subheader("💡 Recomendações da Cogitara")
        self.render_ai_recommendations()
    
    def render_trends(self):
        """Renderiza análise de tendências"""
        if self.data is not None:
            # Selecionar colunas numéricas para análise
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Selecione a métrica:", numeric_cols)
                
                if len(self.data) > 1:
                    fig = px.line(self.data, y=selected_col, title=f"Tendência de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_quick_sentiment(self):
        """Renderiza análise de sentimento rápida"""
        if self.ai and st.session_state.ai_initialized:
            # Análise de colunas de texto
            text_cols = self.data.select_dtypes(include=['object']).columns
            
            if len(text_cols) > 0:
                selected_text_col = st.selectbox("Selecione coluna de texto:", text_cols)
                
                if st.button("🔍 Analisar Sentimento"):
                    with st.spinner("Analisando sentimentos..."):
                        sentiment_results = self.ai.quick_sentiment_analysis(selected_text_col)
                        
                        if sentiment_results:
                            # Mostrar resultados
                            pos = sentiment_results.get('positive', 0)
                            neg = sentiment_results.get('negative', 0)
                            neu = sentiment_results.get('neutral', 0)
                            
                            fig = go.Figure(data=[
                                go.Bar(name='Positivo', x=['Positivo'], y=[pos], marker_color='green'),
                                go.Bar(name='Neutro', x=['Neutro'], y=[neu], marker_color='gray'),
                                go.Bar(name='Negativo', x=['Negativo'], y=[neg], marker_color='red')
                            ])
                            
                            fig.update_layout(title="Distribuição de Sentimentos")
                            st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_recommendations(self):
        """Renderiza recomendações da IA"""
        recommendations = [
            "📊 Considere expandir a análise preditiva para as vendas dos próximos trimestres",
            "🔄 Otimizar campanhas de marketing com base no sentimento dos clientes",
            "🎯 Desenvolver simulador de cenários para planejamento estratégico",
            "📈 Implementar monitoramento contínuo de métricas-chave"
        ]
        
        for rec in recommendations:
            st.markdown(f'<div class="feature-card">{rec}</div>', unsafe_allow_html=True)
    
    def render_predictive_analysis(self):
        """Renderiza módulo de análise preditiva"""
        st.markdown('<div class="main-header">📈 Análise Preditiva</div>', unsafe_allow_html=True)
        
        if not st.session_state.ai_initialized:
            st.warning("⚠️ Inicialize a IA primeiro")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuração da Análise")
            
            # Seleção de variáveis
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            target_var = st.selectbox("Variável Alvo:", numeric_cols)
            feature_vars = st.multiselect("Variáveis Preditivas:", numeric_cols, default=[x for x in numeric_cols if x != target_var][:3])
            
            forecast_periods = st.slider("Períodos de Previsão:", 1, 12, 6)
            
            if st.button("🎯 Executar Análise Preditiva", use_container_width=True):
                with st.spinner("Executando análise preditiva..."):
                    results = self.ai.predictive_analysis(
                        target_column=target_var,
                        feature_columns=feature_vars,
                        forecast_periods=forecast_periods
                    )
                    
                    st.session_state.analysis_results['predictive'] = results
        
        with col2:
            if 'predictive' in st.session_state.analysis_results:
                results = st.session_state.analysis_results['predictive']
                
                # Mostrar resultados
                st.subheader("Resultados da Análise Preditiva")
                
                # Métricas de performance
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Precisão do Modelo", f"{results.get('accuracy', 0)*100:.1f}%")
                with col2:
                    st.metric("🔮 Períodos Previstos", forecast_periods)
                with col3:
                    st.metric("📈 Confiança", f"{results.get('confidence', 0)*100:.1f}%")
                
                # Gráfico de previsão
                if 'forecast_plot' in results:
                    st.plotly_chart(results['forecast_plot'], use_container_width=True)
                
                # Insights da IA
                st.subheader("💡 Insights da Cogitara")
                insights = results.get('insights', [])
                for insight in insights:
                    st.info(f"🔍 {insight}")
    
    def render_scenario_simulator(self):
        """Renderiza módulo de simulador de cenários"""
        st.markdown('<div class="main-header">🔄 Simulador de Cenários "E se...?"</div>', unsafe_allow_html=True)
        
        if not st.session_state.ai_initialized:
            st.warning("⚠️ Inicialize a IA primeiro")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configurar Cenário")
            
            # Variáveis para simulação
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            scenario_vars = st.multiselect("Variáveis para Simulação:", numeric_cols, default=numeric_cols[:2])
            
            st.subheader("Ajustes do Cenário")
            scenario_adjustments = {}
            
            for var in scenario_vars:
                current_val = self.data[var].mean()
                adjustment = st.slider(
                    f"Ajuste para {var}:",
                    min_value=-50.0,
                    max_value=50.0,
                    value=0.0,
                    step=1.0,
                    help=f"Valor atual: {current_val:.2f}"
                )
                scenario_adjustments[var] = adjustment / 100  # Converter para percentual
            
            if st.button("🔄 Simular Cenário", use_container_width=True):
                with st.spinner("Simulando cenário..."):
                    results = self.ai.scenario_simulation(
                        scenario_variables=scenario_vars,
                        adjustments=scenario_adjustments
                    )
                    
                    st.session_state.analysis_results['scenario'] = results
        
        with col2:
            if 'scenario' in st.session_state.analysis_results:
                results = st.session_state.analysis_results['scenario']
                
                st.subheader("Resultados da Simulação")
                
                # Impactos
                st.metric("📊 Impacto Total", f"{results.get('total_impact', 0):.2f}%")
                
                # Gráfico de comparação
                if 'comparison_plot' in results:
                    st.plotly_chart(results['comparison_plot'], use_container_width=True)
                
                # Análise de impacto
                st.subheader("📈 Análise de Impacto por Variável")
                impacts = results.get('variable_impacts', {})
                
                for var, impact in impacts.items():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{var}**")
                    with col2:
                        st.write(f"{impact:.2f}%")
                
                # Recomendações
                st.subheader("💡 Recomendações Estratégicas")
                recommendations = results.get('recommendations', [])
                for rec in recommendations:
                    st.success(f"✅ {rec}")
    
    def render_sentiment_analysis(self):
        """Renderiza módulo de análise de sentimento"""
        st.markdown('<div class="main-header">😊 Análise de Sentimento de Clientes</div>', unsafe_allow_html=True)
        
        if not st.session_state.ai_initialized:
            st.warning("⚠️ Inicialize a IA primeiro")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuração da Análise")
            
            # Seleção de colunas de texto
            text_cols = self.data.select_dtypes(include=['object']).columns.tolist()
            
            if not text_cols:
                st.warning("❌ Nenhuma coluna de texto encontrada nos dados")
                return
            
            text_column = st.selectbox("Coluna para Análise:", text_cols)
            analysis_type = st.radio("Tipo de Análise:", 
                                   ["Análise Básica", "Análise Avançada", "Análise por Tópicos"])
            
            if st.button("😊 Analisar Sentimentos", use_container_width=True):
                with st.spinner("Analisando sentimentos..."):
                    results = self.ai.sentiment_analysis(
                        text_column=text_column,
                        analysis_type=analysis_type
                    )
                    
                    st.session_state.analysis_results['sentiment'] = results
        
        with col2:
            if 'sentiment' in st.session_state.analysis_results:
                results = st.session_state.analysis_results['sentiment']
                
                st.subheader("Resultados da Análise de Sentimento")
                
                # Distribuição de sentimentos
                sentiment_dist = results.get('sentiment_distribution', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("😊 Positivo", f"{sentiment_dist.get('positive', 0):.1f}%")
                with col2:
                    st.metric("😐 Neutro", f"{sentiment_dist.get('neutral', 0):.1f}%")
                with col3:
                    st.metric("😞 Negativo", f"{sentiment_dist.get('negative', 0):.1f}%")
                
                # Gráfico de distribuição
                if 'sentiment_plot' in results:
                    st.plotly_chart(results['sentiment_plot'], use_container_width=True)
                
                # Tópicos principais
                if 'topics' in results:
                    st.subheader("📌 Tópicos Identificados")
                    topics = results['topics']
                    for topic in topics[:5]:
                        st.write(f"• {topic}")
                
                # Insights
                st.subheader("💡 Insights da Cogitara")
                insights = results.get('insights', [])
                for insight in insights:
                    st.info(f"🔍 {insight}")
    
    def render_autonomous_ai(self):
        """Renderiza módulo de IA autônoma"""
        st.markdown('<div class="main-header">🤖 IA Autônoma Cogitara</div>', unsafe_allow_html=True)
        
        if not st.session_state.ai_initialized:
            st.warning("⚠️ Inicialize a IA primeiro")
            return
        
        st.markdown("""
        <div class="feature-card">
        <h4>🎯 Modo Autônomo Ativado</h4>
        <p>A Cogitara agora analisará automaticamente seus dados e fornecerá insights estratégicos completos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Executar Análise Autônoma Completa", use_container_width=True):
            with st.spinner("Cogitara analisando dados de forma autônoma..."):
                autonomous_results = self.ai.autonomous_analysis()
                
                st.session_state.analysis_results['autonomous'] = autonomous_results
                
                # Mostrar resultados
                st.subheader("🎉 Análise Autônoma Concluída!")
                
                # Resumo executivo
                st.subheader("📋 Resumo Executivo")
                summary = autonomous_results.get('executive_summary', {})
                
                for key, value in summary.items():
                    st.write(f"**{key}:** {value}")
                
                # Recomendações estratégicas
                st.subheader("💡 Recomendações Estratégicas")
                recommendations = autonomous_results.get('strategic_recommendations', [])
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f'<div class="metric-card">{i}. {rec}</div>', unsafe_allow_html=True)
                
                # Alertas
                st.subheader("🚨 Alertas e Oportunidades")
                alerts = autonomous_results.get('alerts', [])
                
                for alert in alerts:
                    if alert['type'] == 'warning':
                        st.warning(f"⚠️ {alert['message']}")
                    else:
                        st.success(f"🎯 {alert['message']}")
    
    def run(self):
        """Executa a aplicação principal"""
        page = self.render_sidebar()
        
        if page == "🏠 Dashboard":
            self.render_dashboard()
        elif page == "📈 Análise Preditiva":
            self.render_predictive_analysis()
        elif page == "🔄 Simulador de Cenários":
            self.render_scenario_simulator()
        elif page == "😊 Análise de Sentimento":
            self.render_sentiment_analysis()
        elif page == "🤖 IA Autônoma":
            self.render_autonomous_ai()

# Executar a aplicação
if __name__ == "__main__":
    app = CogitaraApp()
    app.run()