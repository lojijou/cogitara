# modules/ai_core.py
import numpy as np
import pandas as pd

class AICore:
    def __init__(self):
        self.data_patterns = {}

    def learn_data_patterns(self, data: pd.DataFrame):
        numeric = data.select_dtypes(include=['number'])
        for col in numeric.columns:
            s = numeric[col].dropna()
            self.data_patterns[col] = {
                'mean': float(s.mean()) if not s.empty else None,
                'std': float(s.std()) if not s.empty else None,
                'len': int(len(s))
            }

    def quick_summary(self, data: pd.DataFrame):
        numeric = data.select_dtypes(include=['number'])
        summary = {}
        if not numeric.empty:
            for col in numeric.columns[:5]:
                s = numeric[col].dropna()
                summary[f"mean_{col}"] = float(s.mean()) if not s.empty else None
        summary['rows'] = int(len(data))
        summary['columns'] = int(len(data.columns))
        return summary

    def generate_recommendations(self, data):
        recs = []
        numeric = data.select_dtypes(include=['number'])
        if not numeric.empty:
            for col in numeric.columns:
                s = numeric[col].dropna()
                if s.std() / (s.mean() + 1e-9) > 0.5:
                    recs.append(f"Alta variabilidade em {col}. Investigar sazonalidade/anomalias.")
        # default suggestions
        recs.extend([
            "Criar painel com KPIs atualizados.",
            "Implementar monitoramento de anomalias.",
        ])
        return recs

    def detect_alerts(self, data):
        alerts = []
        numeric = data.select_dtypes(include=['number'])
        if not numeric.empty:
            for col in numeric.columns:
                s = numeric[col].dropna()
                if len(s) > 10:
                    q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
                    if iqr > 0:
                        outliers = s[(s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)]
                        if len(outliers) > 0:
                            alerts.append(f"Outliers detectados em {col}: {len(outliers)}")
        return alerts
