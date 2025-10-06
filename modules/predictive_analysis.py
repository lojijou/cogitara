import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

class PredictiveAnalyzer:
    """Módulo de análise preditiva avançada"""
    
    def analyze(self, data, target_column, feature_columns, forecast_periods=6, ai_core=None):
        """Executa análise preditiva robusta"""
        try:
            # Preparar dados
            X = data[feature_columns].copy()
            y = data[target_column].copy()
            
            # Lidar com valores missing
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                return {'error': 'Dados insuficientes para análise (mínimo 10 amostras)'}
            
            # Verificar se é classificação ou regressão
            is_classification = self._check_classification(y)
            
            if is_classification:
                return self._classification_analysis(X, y, feature_columns, target_column)
            else:
                return self._regression_analysis(X, y, feature_columns, target_column, forecast_periods, data)
            
        except Exception as e:
            return {'error': f'Erro na análise preditiva: {str(e)}'}

    def _check_classification(self, y):
        """Verifica se o problema é de classificação"""
        unique_values = y.nunique()
        return unique_values <= 10 and unique_values >= 2

    def _regression_analysis(self, X, y, feature_columns, target_column, forecast_periods, original_data):
        """Análise para problemas de regressão"""
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Treinar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Previsões e métricas
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Previsão futura
        last_data = X.iloc[-1:].values
        forecast = []
        confidence_intervals = []
        
        for i in range(forecast_periods):
            # Prever próximo valor
            next_pred = model.predict(last_data)[0]
            forecast.append(next_pred)
            
            # Calcular intervalo de confiança
            predictions = []
            for tree in model.estimators_:
                pred = tree.predict(last_data)[0]
                predictions.append(pred)
            
            std = np.std(predictions)
            confidence_intervals.append(std)
            
            # Simular dados futuros
            last_data = last_data * (1 + np.random.normal(0.02, 0.01))
        
        # Gráfico
        fig = self._create_forecast_plot(original_data[target_column], forecast, confidence_intervals, target_column)
        
        # Importância das features
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # Insights avançados
        insights = self._generate_regression_insights(model, X, y, feature_importance, r2, mae, forecast)
        
        return {
            'accuracy': max(0, r2),
            'mae': mae,
            'forecast': forecast,
            'confidence_intervals': confidence_intervals,
            'feature_importance': feature_importance,
            'forecast_plot': fig,
            'insights': insights,
            'model_type': 'regressão',
            'confidence': min(0.95, max(0, r2) + 0.1)
        }

    def _classification_analysis(self, X, y, feature_columns, target_column):
        """Análise para problemas de classificação"""
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Treinar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Previsões e métricas
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Importância das features
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        # Matriz de confusão (simplificada)
        confusion_info = self._get_confusion_info(y_test, y_pred)
        
        # Insights
        insights = self._generate_classification_insights(model, X, y, feature_importance, accuracy, confusion_info)
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'confusion_info': confusion_info,
            'insights': insights,
            'model_type': 'classificação',
            'confidence': min(0.95, accuracy + 0.1)
        }

    def _create_forecast_plot(self, historical, forecast, confidence_intervals, target_name):
        """Cria gráfico de previsão com intervalo de confiança"""
        historical_periods = list(range(len(historical)))
        forecast_periods = list(range(len(historical), len(historical) + len(forecast)))
        
        fig = go.Figure()
        
        # Histórico
        fig.add_trace(go.Scatter(
            x=historical_periods,
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
        
        # Intervalo de confiança
        upper_bound = [historical.iloc[-1]] + [forecast[i] + confidence_intervals[i] for i in range(len(forecast))]
        lower_bound = [historical.iloc[-1]] + [forecast[i] - confidence_intervals[i] for i in range(len(forecast))]
        
        fig.add_trace(go.Scatter(
            x=forecast_periods,
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_periods,
            y=lower_bound,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"Previsão - {target_name}",
            xaxis_title="Períodos",
            yaxis_title=target_name,
            showlegend=True
        )
        
        return fig

    def _get_confusion_info(self, y_true, y_pred):
        """Informações da matriz de confusão"""
        from sklearn.metrics import classification_report, confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }

    def _generate_regression_insights(self, model, X, y, feature_importance, r2, mae, forecast):
        """Gera insights para regressão"""
        insights = []
        
        # Insight de precisão
        if r2 > 0.8:
            insights.append("🎯 Modelo altamente preciso - confiável para decisões")
        elif r2 > 0.6:
            insights.append("📈 Boa precisão - adequado para planejamento")
        else:
            insights.append("⚠️ Precisão moderada - considere mais variáveis")
        
        # Insight de tendência
        trend = "positiva" if np.mean(forecast) > y.iloc[-1] else "negativa"
        insights.append(f"📊 Tendência {trend} identificada")
        
        # Insight de variáveis importantes
        top_feature = max(feature_importance, key=feature_importance.get)
        insights.append(f"🔍 Variável mais influente: {top_feature}")
        
        # Insight de erro
        insights.append(f"📏 Erro médio de {mae:.2f} unidades")
        
        return insights

    def _generate_classification_insights(self, model, X, y, feature_importance, accuracy, confusion_info):
        """Gera insights para classificação"""
        insights = []
        
        # Insight de precisão
        if accuracy > 0.9:
            insights.append("🎯 Classificação excelente - altamente confiável")
        elif accuracy > 0.7:
            insights.append("📊 Boa precisão - adequada para uso")
        else:
            insights.append("⚠️ Precisão moderada - avalie cuidadosamente")
        
        # Insight de variáveis importantes
        top_feature = max(feature_importance, key=feature_importance.get)
        insights.append(f"🔍 Fator mais determinante: {top_feature}")
        
        # Insight de métricas
        insights.append(f"📈 Precisão: {accuracy:.1%}")
        insights.append(f"🎯 F1-Score: {confusion_info['f1_score']:.1%}")
        
        return insights

    def quick_analyze(self, data, target_column):
        """Análise rápida automática"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        if len(feature_cols) == 0:
            return {'error': 'Variáveis insuficientes para análise'}
        
        return self.analyze(data, target_column, feature_cols[:3], 3)
