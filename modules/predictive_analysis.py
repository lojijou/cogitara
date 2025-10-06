import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import plotly.graph_objects as go

class PredictiveAnalyzer:
    def analyze(self, data, target_column, feature_columns, forecast_periods=6, ai_core=None):
        """Executa análise preditiva"""
        try:
            # Preparar dados
            X = data[feature_columns]
            y = data[target_column]
            
            # Remover NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                return {'error': 'Dados insuficientes para análise'}
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Treinar modelo
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Previsões
            y_pred = model.predict(X_test)
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            accuracy = max(0, r2)
            
            # Previsão futura
            last_values = X.iloc[-1:].values
            forecast = []
            for _ in range(forecast_periods):
                pred = model.predict(last_values)[0]
                forecast.append(pred)
                last_values = last_values * 1.02
            
            # Gráfico
            fig = self._create_forecast_plot(data[target_column], forecast, target_column)
            
            return {
                'accuracy': accuracy,
                'confidence': min(0.95, accuracy + 0.1),
                'mae': mae,
                'forecast': forecast,
                'forecast_plot': fig,
                'insights': [
                    f"Modelo com {accuracy*100:.1f}% de precisão",
                    f"Tendência {'positiva' if np.mean(forecast) > y.iloc[-1] else 'negativa'} identificada",
                    f"Recomenda-se monitorar {target_column} nos próximos {forecast_periods} períodos"
                ]
            }
            
        except Exception as e:
            return {'error': f'Erro na análise preditiva: {str(e)}'}

    def _create_forecast_plot(self, historical, forecast, target_name):
        """Cria gráfico de previsão"""
        periods = list(range(1, len(historical) + 1))
        forecast_periods = list(range(len(historical), len(historical) + len(forecast)))
        
        fig = go.Figure()
        
        # Histórico
        fig.add_trace(go.Scatter(
            x=periods,
            y=historical.values,
            mode='lines+markers',
            name='Histórico',
            line=dict(color='blue', width=3)
        ))
        
        # Previsão
        fig.add_trace(go.Scatter(
            x=forecast_periods,
            y=[historical.iloc[-1]] + forecast,
            mode='lines+markers',
            name='Previsão',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title=f"Previsão para {target_name}",
            xaxis_title="Períodos",
            yaxis_title=target_name
        )
        
        return fig
