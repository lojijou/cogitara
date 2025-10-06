# modules/predictive_analysis.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

class PredictiveAnalyzer:
    def analyze(self, data: pd.DataFrame, target_column: str, feature_columns=None, forecast_periods=6, ai_core=None):
        try:
            df = data.dropna(subset=[target_column])
            y = df[target_column].values
            # simple time index model: regress y ~ t (fast fallback)
            t = np.arange(len(y)).reshape(-1,1)
            model = LinearRegression().fit(t, y)
            future_t = np.arange(len(y), len(y)+forecast_periods).reshape(-1,1)
            preds = model.predict(future_t)

            # Build plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y, mode='lines+markers', name='histórico'))
            fig.add_trace(go.Scatter(x=np.arange(len(y), len(y)+forecast_periods), y=preds, mode='lines+markers', name='previsão'))
            fig.update_layout(title=f"Previsão simples para {target_column}", xaxis_title="periodo", yaxis_title=target_column)

            # table of forecast
            forecast_table = pd.DataFrame({
                'period': np.arange(len(y), len(y)+forecast_periods),
                'prediction': preds
            })

            insights = [
                f"Modelo linear simples treinado sobre {len(y)} pontos",
                f"Última previsão: {preds[-1]:.2f}"
            ]

            return {'forecast_plot': fig, 'forecast_table': forecast_table.to_dict(orient='records'), 'insights': insights}
        except Exception as e:
            return {'error': str(e)}
