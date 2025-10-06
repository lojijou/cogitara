import pandas as pd
import numpy as np
import warnings
import streamlit as st  # Corrigido: import do Streamlit
warnings.filterwarnings('ignore')

# Importações para módulos
try:
    from modules.predictive_analysis import PredictiveAnalyzer
    from modules.scenario_simulator import ScenarioSimulator
    from modules.sentiment_analysis import SentimentAnalyzer
    from modules.ai_core import AICore
except ImportError:
    try:
        from .modules.predictive_analysis import PredictiveAnalyzer
        from .modules.scenario_simulator import ScenarioSimulator
        from .modules.sentiment_analysis import SentimentAnalyzer
        from .modules.ai_core import AICore
    except ImportError:
        # Classes fallback
        class PredictiveAnalyzer:
            def analyze(self, data, target_column, feature_columns=None, forecast_periods=6, ai_core=None):
                return {'error': 'Módulo preditivo não carregado'}
        
        class ScenarioSimulator:
            def simulate(self, data, scenario_variables, adjustments, ai_core=None):
                return {'error': 'Módulo de simulação não carregado'}
        
        class SentimentAnalyzer:
            def analyze(self, data, text_column, analysis_type="Análise Básica", ai_core=None):
                return {'error': 'Módulo de sentimento não carregado'}
            def quick_analyze(self, data, text_column):
                return {'error': 'Módulo de sentimento não carregado'}
        
        class AICore:
            def learn_data_patterns(self, data):
                pass

class CogitaraAI:
    """Classe principal da IA Cogitara - IA Empresarial Autônoma"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.ai_core = AICore()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.scenario_simulator = ScenarioSimulator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self._initialize_ai()
    
    def _initialize_ai(self):
        """Inicializa os componentes da IA"""
        st.write("🚀 Inicializando Cogitara AI...")
        try:
            self.ai_core.learn_data_patterns(self.data)
            st.write("✅ Cogitara AI inicializada com sucesso!")
        except Exception as e:
            st.error(f"Erro ao inicializar AI: {e}")
    
    def predictive_analysis(self, target_column, feature_columns=None, forecast_periods=6):
        """Executa análise preditiva"""
        if feature_columns is None:
            feature_columns = [col for col in self.data.select_dtypes(include=[np.number]).columns 
                               if col != target_column]
        
        if not feature_columns:
            st.warning("Nenhuma coluna de recurso disponível para previsão.")
        
        try:
            return self.predictive_analyzer.analyze(
                data=self.data,
                target_column=target_column,
                feature_columns=feature_columns,
                forecast_periods=forecast_periods,
                ai_core=self.ai_core
            )
        except Exception as e:
            st.error(f"Erro na análise preditiva: {e}")
            return {}
    
    def scenario_simulation(self, scenario_variables, adjustments):
        """Executa simulação de cenários"""
        try:
            return self.scenario_simulator.simulate(
                data=self.data,
                scenario_variables=scenario_variables,
                adjustments=adjustments,
                ai_core=self.ai_core
            )
        except Exception as e:
            st.error(f"Erro na simulação de cenários: {e}")
            return {}
    
    def sentiment_analysis(self, text_column, analysis_type="Análise Básica"):
        """Executa análise de sentimento"""
        try:
            return self.sentiment_analyzer.analyze(
                data=self.data,
                text_column=text_column,
                analysis_type=analysis_type,
                ai_core=self.ai_core
            )
        except Exception as e:
            st.error(f"Erro na análise de sentimento: {e}")
            return {}
    
    def quick_sentiment_analysis(self, text_column):
        """Análise rápida de sentimento"""
        try:
            return self.sentiment_analyzer.quick_analyze(
                data=self.data,
                text_column=text_column
            )
        except Exception as e:
            st.error(f"Erro na análise rápida de sentimento: {e}")
            return {}
    
    def autonomous_analysis(self):
        """Executa análise autônoma completa"""
        st.write("🤖 Iniciando análise autônoma...")
        try:
            results = {
                'executive_summary': self._generate_executive_summary(),
                'strategic_recommendations': self._generate_strategic_recommendations(),
                'alerts': self._generate_alerts(),
                'key_insights': self._generate_key_insights()
            }
            return results
        except Exception as e:
            st.error(f"Erro na análise autônoma: {e}")
            return {}
    
    def _generate_executive_summary(self):
        """Gera resumo executivo automático"""
        summary = {}
        numeric_data = self.data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            summary['Total de Registros'] = len(self.data)
            summary['Variáveis Numéricas'] = len(numeric_data.columns)
            summary['Período de Dados'] = "Personalizado"
            for col in numeric_data.columns[:3]:
                try:
                    summary[f'Média de {col}'] = numeric_data[col].mean()
                except Exception:
                    summary[f'Média de {col}'] = None
        return summary
    
    def _generate_strategic_recommendations(self):
        """Gera recomendações estratégicas automáticas"""
        recommendations = []
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            # Recomendações baseadas em variabilidade
            for col in numeric_data.columns:
                try:
                    mean_val = numeric_data[col].mean()
                    std_val = numeric_data[col].std()
                    if mean_val != 0 and (std_val / mean_val) > 0.5:
                        recommendations.append(
                            f"Alta variabilidade em {col}. Considere investigar causas."
                        )
                except Exception:
                    continue
            
            # Recomendações baseadas em correlações
            if len(numeric_data.columns) >= 2:
                try:
                    corr_matrix = numeric_data.corr()
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr = corr_matrix.iloc[i, j]
                            if abs(corr) > 0.7:
                                recommendations.append(
                                    f"Forte correlação ({corr:.2f}) entre {corr_matrix.columns[i]} e {corr_matrix.columns[j]}."
                                )
                except Exception:
                    pass
        
        # Recomendações padrão
        default_recommendations = [
            "Implementar monitoramento contínuo das métricas-chave",
            "Desenvolver dashboard executivo para acompanhamento",
            "Criar sistema de alertas para anomalias",
            "Expandir análise com dados externos"
        ]
        recommendations.extend(default_recommendations)
        return recommendations[:6]
    
    def _generate_alerts(self):
        """Gera alertas automáticos"""
        alerts = []
        numeric_data = self.data.select_dtypes(include=[np.number])
        for col in numeric_data.columns:
            try:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = numeric_data[(numeric_data[col] < (Q1 - 1.5*IQR)) | 
                                        (numeric_data[col] > (Q3 + 1.5*IQR))]
                if len(outliers) > 0:
                    alerts.append({
                        'type': 'warning',
                        'message': f"Outliers em {col} ({len(outliers)} ocorrências)"
                    })
            except Exception:
                continue
        return alerts
    
    def _generate_key_insights(self):
        """Gera insights chave dos dados"""
        insights = []
        numeric_data = self.data.select_dtypes(include=[np.number])
        for col in numeric_data.columns:
            try:
                skewness = numeric_data[col].skew()
                if abs(skewness) > 1:
                    insights.append(f"Distribuição assimétrica em {col}")
            except Exception:
                continue
        return insights
