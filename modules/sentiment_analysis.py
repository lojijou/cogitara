import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Módulo avançado de análise de sentimento"""
    
    def __init__(self):
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.intensity_modifiers = self._load_intensity_modifiers()
    
    def analyze(self, data, text_column, analysis_type="Análise Básica", ai_core=None):
        """Análise de sentimento completa"""
        try:
            texts = data[text_column].dropna().astype(str)
            
            if len(texts) == 0:
                return {'error': 'Nenhum texto válido para análise'}
            
            # Análise de sentimento
            sentiment_results = self._analyze_sentiments(texts)
            
            # Análise de tópicos
            topics_analysis = self._analyze_topics(texts)
            
            # Análise de emoções
            emotions_analysis = self._analyze_emotions(texts)
            
            # Métricas de engajamento
            engagement_metrics = self._calculate_engagement_metrics(texts)
            
            # Gráficos
            sentiment_plot = self._create_sentiment_plot(sentiment_results['distribution'])
            topics_plot = self._create_topics_plot(topics_analysis['top_topics'])
            emotions_plot = self._create_emotions_plot(emotions_analysis)
            
            # Insights estratégicos
            insights = self._generate_strategic_insights(
                sentiment_results, topics_analysis, emotions_analysis, engagement_metrics
            )
            
            return {
                'sentiment_distribution': sentiment_results['distribution'],
                'sentiment_scores': sentiment_results['scores'],
                'topics_analysis': topics_analysis,
                'emotions_analysis': emotions_analysis,
                'engagement_metrics': engagement_metrics,
                'sentiment_plot': sentiment_plot,
                'topics_plot': topics_plot,
                'emotions_plot': emotions_plot,
                'insights': insights,
                'sample_size': len(sentiment_results['scores']),
                'analysis_type': analysis_type
            }
            
        except Exception as e:
            return {'error': f'Erro na análise de sentimento: {str(e)}'}

    def _load_positive_words(self):
        """Lista de palavras positivas em português"""
        return {
            'bom', 'boa', 'excelente', 'ótimo', 'ótima', 'maravilhoso', 'maravilhosa',
            'incrível', 'fantástico', 'perfeito', 'perfeita', 'gostei', 'adoro', 'amo',
            'recomendo', 'eficiente', 'rápido', 'rápida', 'qualidade', 'atendimento',
            'solícito', 'solícita', 'educado', 'educada', 'prestativo', 'prestativa',
            'resolutivo', 'resolutiva', 'satisfeito', 'satisfeita', 'feliz', 'content
