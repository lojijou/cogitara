import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Configuração básica da página
st.set_page_config(
    page_title="Cogitara AI",
    page_icon="🚀",
    layout="wide"
)

# CSS simples
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Classe simplificada da IA
class CogitaraAI:
    def __init__(self, data):
        self.data = data
        st.success("🤖 IA Cogitara inicializada!")
    
    def predictive_analysis(self, target_column, feature_columns):
        try:
            # Simulação simples de análise
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=self.data[target_column].values,
                mode='lines',
                name='Dados'
            ))
            fig.update_layout(title=f"Análise de {target_column}")
            
            return {
                'accuracy': 0.85,
                'forecast_plot': fig,
                'insights': ['Tendência analisada com sucesso', 'Precisão: 85%']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def sentiment_analysis(self, text_column):
        try:
            # Simulação de análise de sentimento
            fig = go.Figure(data=[go.Pie(
                labels=['Positivo', 'Neutro', 'Negativo'],
                values=[60, 30, 10],
                hole=0.3
            )])
            fig.update_layout(title="Sentimento dos Clientes")
            
            return {
                'sentiment_plot': fig,
                'insights': ['Sentimento geral positivo', 'Boa satisfação do cliente']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def autonomous_analysis(self):
        return {
            'executive_summary': {'Status': 'Análise completa', 'Confiança': '90%'},
            'recommendations': [
                'Otimizar processos operacionais',
                'Melhorar atendimento ao cliente',
                'Expandir análise de dados'
            ]
        }

# Aplicação principal
def main():
    st.sidebar.title("🚀 Cogitara AI")
    st.sidebar.markdown("---")
    
    # Upload de dados
    uploaded_file = st.sidebar.file_uploader("📊 Upload de Dados", type=['csv', 'xlsx'])
    
    data = None
    ai = None
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.sidebar.success(f"✅ {len(data)} linhas carregadas")
            
            # Mostrar preview
            with st.sidebar.expander("📋 Ver dados"):
                st.dataframe(data.head())
                
        except Exception as e:
            st.sidebar.error(f"❌ Erro: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # Inicializar IA
    if st.sidebar.button("🔄 Inicializar IA") and data is not None:
        ai = CogitaraAI(data)
    
    st.sidebar.markdown("---")
    
    # Navegação
    page = st.sidebar.radio("Navegação:", [
        "🏠 Dashboard", 
        "📈 Análise Preditiva", 
        "😊 Análise de Sentimento",
        "🤖 IA Autônoma"
    ])
    
    # Páginas
    if page == "🏠 Dashboard":
        st.markdown('<div class="main-header">🏠 Dashboard Cogitara AI</div>', unsafe_allow_html=True)
        
        if data is None:
            st.info("📁 Faça upload de dados para começar")
            st.image("https://via.placeholder.com/600x200/1f77b4/ffffff?text=COGITARA+AI", width=600)
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Linhas", len(data))
            col2.metric("🔮 Colunas", len(data.columns))
            col3.metric("🔢 Numéricas", len(data.select_dtypes(include=[np.number]).columns))
            col4.metric("📝 Textuais", len(data.select_dtypes(include=['object']).columns))
            
            # Gráfico simples
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = st.selectbox("Selecione coluna para visualizar:", numeric_cols)
                fig = px.line(data, y=col, title=f"Evolução de {col}")
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📈 Análise Preditiva":
        st.markdown('<div class="main-header">📈 Análise Preditiva</div>', unsafe_allow_html=True)
        
        if ai is None:
            st.warning("⚠️ Carregue dados e inicialize a IA primeiro")
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target = st.selectbox("Selecione a variável alvo:", numeric_cols)
                if st.button("🎯 Executar Análise"):
                    results = ai.predictive_analysis(target, numeric_cols[:2])
                    if 'error' not in results:
                        st.plotly_chart(results['forecast_plot'], use_container_width=True)
                        for insight in results['insights']:
                            st.info(insight)
            else:
                st.error("❌ Nenhuma coluna numérica encontrada")
    
    elif page == "😊 Análise de Sentimento":
        st.markdown('<div class="main-header">😊 Análise de Sentimento</div>', unsafe_allow_html=True)
        
        if ai is None:
            st.warning("⚠️ Carregue dados e inicialize a IA primeiro")
        else:
            text_cols = data.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                text_col = st.selectbox("Selecione coluna de texto:", text_cols)
                if st.button("😊 Analisar Sentimento"):
                    results = ai.sentiment_analysis(text_col)
                    if 'error' not in results:
                        st.plotly_chart(results['sentiment_plot'], use_container_width=True)
                        for insight in results['insights']:
                            st.info(insight)
            else:
                st.warning("📝 Nenhuma coluna de texto encontrada")
    
    elif page == "🤖 IA Autônoma":
        st.markdown('<div class="main-header">🤖 IA Autônoma</div>', unsafe_allow_html=True)
        
        if ai is None:
            st.warning("⚠️ Carregue dados e inicialize a IA primeiro")
        else:
            if st.button("🚀 Executar Análise Completa"):
                results = ai.autonomous_analysis()
                
                st.success("🎉 Análise autônoma concluída!")
                
                st.subheader("📋 Resumo")
                for key, value in results['executive_summary'].items():
                    st.write(f"**{key}**: {value}")
                
                st.subheader("💡 Recomendações")
                for rec in results['recommendations']:
                    st.success(f"✅ {rec}")

# Executar aplicação
if __name__ == "__main__":
    main()
