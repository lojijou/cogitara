import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class AICore:
    """Núcleo de inteligência artificial da Cogitara"""
    
    def __init__(self):
        self.data_patterns = {}
        self.ml_models = {}
        self.text_models = {}
        self.learned_insights = []
    
    def learn_data_patterns(self, data):
        """Aprende padrões dos dados automaticamente"""
        print("🧠 Aprendendo padrões dos dados...")
        
        # Análise de dados numéricos
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            self._learn_numeric_patterns(numeric_data)
        
        # Análise de dados textuais
        text_data = data.select_dtypes(include=['object'])
        if not text_data.empty:
            self._learn_text_patterns(text_data)
        
        print(f"✅ Padrões aprendidos: {len(self.data_patterns)} insights")
    
    def _learn_numeric_patterns(self, numeric_data):
        """Aprende padrões de dados numéricos"""
        for col in numeric_data.columns:
            self.data_patterns[f"{col}_stats"] = {
                'mean': numeric_data[col].mean(),
                'std': numeric_data[col].std(),
                'trend': self._calculate_trend(numeric_data[col])
            }
    
    def _learn_text_patterns(self, text_data):
        """Aprende padrões de dados textuais"""
        for col in text_data.columns:
            sample_texts = text_data[col].dropna().head(100)
            if len(sample_texts) > 0:
                unique_ratio = len(sample_texts.unique()) / len(sample_texts)
                self.data_patterns[f"{col}_text_stats"] = {
                    'unique_ratio': unique_ratio,
                    'avg_length': sample_texts.str.len().mean()
                }
    
    def _calculate_trend(self, series):
        """Calcula tendência de uma série temporal"""
        if len(series) > 1:
            x = np.arange(len(series))
            y = series.values
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0
    
    def generate_business_insights(self, data):
        """Gera insights de negócio automaticamente"""
        insights = []
        
        # Insights baseados em correlações
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) >= 2:
            corr_matrix = numeric_data.corr()
            # Encontrar correlações fortes
            # ... implementação detalhada ...
        
        return insights