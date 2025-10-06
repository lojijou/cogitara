# modules/sentiment_analysis.py
import pandas as pd
import plotly.graph_objects as go
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER resources
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self._sia = SentimentIntensityAnalyzer()

    def analyze(self, data: pd.DataFrame, text_column: str, analysis_type="basic", ai_core=None):
        try:
            texts = data[text_column].dropna().astype(str)
            if texts.empty:
                return {'error': 'Nenhum texto válido'}
            scores = texts.apply(lambda t: self._sia.polarity_scores(t)['compound'])
            positive = (scores > 0.05).sum()
            neutral = ((scores >= -0.05) & (scores <= 0.05)).sum()
            negative = (scores < -0.05).sum()
            total = len(scores)
            dist = {'positive': float(positive/total*100), 'neutral': float(neutral/total*100), 'negative': float(negative/total*100)}
            fig = go.Figure(data=[go.Pie(labels=list(dist.keys()), values=list(dist.values()), hole=0.3)])
            fig.update_layout(title="Distribuição de Sentimentos (%)")
            insights = [f"{dist['positive']:.1f}% positivo — atenção a {negative} comentários negativos."]
            topics = []  # placeholder for future topic extraction
            return {'sentiment_distribution': dist, 'sentiment_plot': fig, 'insights': insights, 'topics': topics}
        except Exception as e:
            return {'error': str(e)}

    def quick_analyze(self, data, text_column):
        try:
            texts = data[text_column].dropna().astype(str)
            if texts.empty:
                return {'error': 'Nenhum texto'}
            scores = texts.apply(lambda t: self._sia.polarity_scores(t)['compound'])
            positive = (scores > 0.05).sum()
            neutral = ((scores >= -0.05) & (scores <= 0.05)).sum()
            negative = (scores < -0.05).sum()
            total = len(scores)
            return {'positive': float(positive/total*100), 'neutral': float(neutral/total*100), 'negative': float(negative/total*100)}
        except Exception as e:
            return {'error': str(e)}
