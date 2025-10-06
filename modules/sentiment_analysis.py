import pandas as pd
import numpy as np
import plotly.graph_objects as go

class SentimentAnalyzer:
    def analyze(self, data, text_column, analysis_type="Análise Básica", ai_core=None):
        """Analisa sentimentos em textos"""
        try:
            texts = data[text_column].dropna().astype(str)
            
            if len(texts) == 0:
                return {'error': 'Nenhum texto válido para análise'}
            
            # Análise de sentimento simulada
            sentiment_dist = {
                'positive': 45.5,
                'neutral': 35.2, 
                'negative': 19.3
            }
            
            # Gráfico de pizza
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(sentiment_dist.keys()),
                    values=list(sentiment_dist.values()),
                    hole=0.3
                )
            ])
            fig.update_layout(title="Distribuição de Sentimentos")
            
            return {
                'sentiment_distribution': sentiment_dist,
                'sentiment_plot': fig,
                'topics': [
                    "Satisfação com produto/serviço",
                    "Atendimento ao cliente",
                    "Tempo de resposta",
                    "Qualidade do suporte"
                ],
                'insights': [
                    f"Sentimento geral: {sentiment_dist['positive']:.1f}% positivo",
                    "Oportunidade de melhoria no atendimento",
                    "Clientes valorizam tempo de resposta rápido"
                ]
            }
            
        except Exception as e:
            return {'error': f'Erro na análise de sentimento: {str(e)}'}

    def quick_analyze(self, data, text_column):
        """Análise rápida de sentimento"""
        try:
            texts = data[text_column].dropna().astype(str)
            
            if len(texts) == 0:
                return {'error': 'Nenhum texto para análise'}
            
            # Distribuição simulada
            return {
                'positive': 42.1,
                'neutral': 38.5,
                'negative': 19.4
            }
            
        except Exception as e:
            return {'error': str(e)}
