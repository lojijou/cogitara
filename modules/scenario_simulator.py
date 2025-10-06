import pandas as pd
import numpy as np
import plotly.graph_objects as go

class ScenarioSimulator:
    def simulate(self, data, scenario_variables, adjustments, ai_core=None):
        """Simula cenários 'E se...?'"""
        try:
            impacts = {}
            total_impact = 0
            
            for var, adjustment in adjustments.items():
                if var in data.columns:
                    current_mean = data[var].mean()
                    new_mean = current_mean * (1 + adjustment)
                    impact = ((new_mean - current_mean) / current_mean) * 100
                    impacts[var] = impact
                    total_impact += abs(impact)
            
            # Gráfico de comparação
            fig = self._create_comparison_plot(impacts)
            
            # Recomendações
            recommendations = []
            if total_impact > 0:
                recommendations = [
                    "Cenário favorável identificado",
                    "Considere implementar os ajustes propostos",
                    "Monitore os resultados em tempo real"
                ]
            else:
                recommendations = [
                    "Cenário requer cautela",
                    "Analise impactos negativos antes de implementar"
                ]
            
            return {
                'total_impact': total_impact / len(adjustments) if adjustments else 0,
                'variable_impacts': impacts,
                'comparison_plot': fig,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {'error': f'Erro na simulação: {str(e)}'}

    def _create_comparison_plot(self, impacts):
        """Cria gráfico de comparação"""
        variables = list(impacts.keys())
        values = list(impacts.values())
        
        colors = ['green' if x > 0 else 'red' for x in values]
        
        fig = go.Figure(data=[
            go.Bar(x=variables, y=values, marker_color=colors)
        ])
        
        fig.update_layout(
            title="Impacto das Variáveis no Cenário",
            xaxis_title="Variáveis",
            yaxis_title="Impacto (%)"
        )
        
        return fig
