import pandas as pd
import numpy as np

class AICore:
    def __init__(self):
        self.data_patterns = {}
    
    def learn_data_patterns(self, data):
        """Aprende padrões dos dados"""
        # Análise de dados numéricos
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            for col in numeric_data.columns:
                self.data_patterns[f"{col}_stats"] = {
                    'mean': numeric_data[col].mean(),
                    'std': numeric_data[col].std(),
                    'trend': self._calculate_trend(numeric_data[col])
                }

    def _calculate_trend(self, series):
        """Calcula tendência de uma série"""
        if len(series) > 1:
            x = np.arange(len(series))
            y = series.values
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope
            except:
                return 0
        return 0
