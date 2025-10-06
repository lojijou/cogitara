import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ScenarioSimulator:
    """Módulo de simulação de cenários 'E se...?'"""
    
    def simulate(self, data, scenario_variables, adjustments, ai_core=None):
        """Simula cenários com análise de impacto multivariada"""
        try:
            # Validar entradas
            if not scenario_variables:
                return {'error': 'Nenhuma variável selecionada para simulação'}
            
            if not adjustments:
                return {'error': 'Nenhum ajuste definido para simulação'}
            
            # Análise de impacto individual
            individual_impacts = self._calculate_individual_impacts(data, adjustments)
            
            # Análise de impacto combinado
            combined_impact = self._calculate_combined_impact(data, scenario_variables, adjustments)
            
            # Análise de sensibilidade
            sensitivity_analysis = self._sensitivity_analysis(data, scenario_variables, adjustments)
            
            # Gráficos
            comparison_plot = self._create_comparison_plot(individual_impacts)
            sensitivity_plot = self._create_sensitivity_plot(sensitivity_analysis)
            
            # Recomendações estratégicas
            recommendations = self._generate_strategic_recommendations(
                individual_impacts, combined_impact, sensitivity_analysis
            )
            
            # Análise de risco
            risk_assessment = self._assess_risk(individual_impacts, combined_impact)
            
            return {
                'individual_impacts': individual_impacts,
                'combined_impact': combined_impact,
                'sensitivity_analysis': sensitivity_analysis,
                'risk_assessment': risk_assessment,
                'comparison_plot': comparison_plot,
                'sensitivity_plot': sensitivity_plot,
                'recommendations': recommendations,
                'scenario_variables': scenario_variables,
                'adjustments_applied': adjustments
            }
            
        except Exception as e:
            return {'error': f'Erro na simulação de cenários: {str(e)}'}

    def _calculate_individual_impacts(self, data, adjustments):
        """Calcula impacto individual de cada ajuste"""
        impacts = {}
        
        for var, adjustment in adjustments.items():
            if var in data.columns and data[var].dtype in [np.number]:
                current_stats = self._get_variable_stats(data[var])
                new_value = current_stats['mean'] * (1 + adjustment)
                
                impact_percent = ((new_value - current_stats['mean']) / current_stats['mean']) * 100
                volatility_impact = abs(impact_percent) * current_stats['volatility']
                
                impacts[var] = {
                    'impact_percent': impact_percent,
                    'volatility_impact': volatility_impact,
                    'current_value': current_stats['mean'],
                    'new_value': new_value,
                    'magnitude': abs(impact_percent),
                    'direction': 'positive' if impact_percent > 0 else 'negative'
                }
        
        return impacts

    def _calculate_combined_impact(self, data, scenario_variables, adjustments):
        """Calcula impacto combinado considerando correlações"""
        try:
            # Selecionar apenas variáveis numéricas
            numeric_data = data[scenario_variables].select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                return {'total_impact': 0, 'interaction_effects': {}}
            
            # Calcular matriz de correlação
            corr_matrix = numeric_data.corr()
            
            # Calcular impacto base
            base_impact = 0
            interaction_effects = {}
            
            for i, var1 in enumerate(scenario_variables):
                if var1 in adjustments:
                    adjustment1 = adjustments[var1]
                    impact1 = self._calculate_individual_impacts(data, {var1: adjustment1})[var1]['impact_percent']
                    base_impact += impact1
                    
                    # Calcular efeitos de interação
                    for j, var2 in enumerate(scenario_variables[i+1:], i+1):
                        if var2 in adjustments:
                            adjustment2 = adjustments[var2]
                            impact2 = self._calculate_individual_impacts(data, {var2: adjustment2})[var2]['impact_percent']
                            
                            # Efeito de interação baseado na correlação
                            if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
                                correlation = corr_matrix.loc[var1, var2]
                                interaction_effect = impact1 * impact2 * correlation * 0.1  # Fator de escala
                                interaction_effects[f"{var1}_{var2}"] = interaction_effect
                                base_impact += interaction_effect
            
            total_impact = base_impact / len(adjustments) if adjustments else 0
            
            return {
                'total_impact': total_impact,
                'interaction_effects': interaction_effects,
                'adjusted_variables_count': len(adjustments)
            }
            
        except Exception as e:
            return {'total_impact': 0, 'interaction_effects': {}, 'error': str(e)}

    def _sensitivity_analysis(self, data, scenario_variables, base_adjustments):
        """Análise de sensibilidade dos parâmetros"""
        sensitivity_results = {}
        
        for var in scenario_variables:
            if var in base_adjustments:
                base_adj = base_adjustments[var]
                
                # Testar variações de ±25%
                variations = [-0.25, -0.1, 0, 0.1, 0.25]
                impacts = []
                
                for variation in variations:
                    test_adjustment = base_adj * (1 + variation)
                    test_impacts = self._calculate_individual_impacts(
                        data, {var: test_adjustment}
                    )
                    impacts.append(test_impacts[var]['impact_percent'])
                
                sensitivity_results[var] = {
                    'base_impact': impacts[2],  # Impacto no ajuste base
                    'sensitivity_range': max(impacts) - min(impacts),
                    'max_impact': max(impacts),
                    'min_impact': min(impacts),
                    'elasticity': (max(impacts) - min(impacts)) / (0.5 if base_adj != 0 else 1)  # Evitar divisão por zero
                }
        
        return sensitivity_results

    def _get_variable_stats(self, series):
        """Calcula estatísticas da variável"""
        return {
            'mean': series.mean(),
            'std': series.std(),
            'volatility': series.std() / series.mean() if series.mean() != 0 else 0,
            'min': series.min(),
            'max': series.max()
        }

    def _create_comparison_plot(self, individual_impacts):
        """Cria gráfico de comparação de impactos"""
        variables = list(individual_impacts.keys())
        impacts = [individual_impacts[var]['impact_percent'] for var in variables]
        
        colors = ['green' if impact > 0 else 'red' for impact in impacts]
        
        fig = go.Figure(data=[
            go.Bar(
                x=variables, 
                y=impacts, 
                marker_color=colors,
                text=[f"{impact:.1f}%" for impact in impacts],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Impacto Individual das Variáveis",
            xaxis_title="Variáveis",
            yaxis_title="Impacto (%)",
            showlegend=False
        )
        
        return fig

    def _create_sensitivity_plot(self, sensitivity_analysis):
        """Cria gráfico de análise de sensibilidade"""
        if not sensitivity_analysis:
            return None
            
        variables = list(sensitivity_analysis.keys())
        sensitivities = [sensitivity_analysis[var]['sensitivity_range'] for var in variables]
        
        fig = go.Figure(data=[
            go.Bar(
                x=variables, 
                y=sensitivities,
                marker_color='orange',
                text=[f"{sens:.1f}%" for sens in sensitivities],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Análise de Sensibilidade das Variáveis",
            xaxis_title="Variáveis",
            yaxis_title="Variação do Impacto (%)",
            showlegend=False
        )
        
        return fig

    def _generate_strategic_recommendations(self, individual_impacts, combined_impact, sensitivity_analysis):
        """Gera recomendações estratégicas baseadas na simulação"""
        recommendations = []
        
        # Análise de impacto geral
        total_impact = combined_impact.get('total_impact', 0)
        
        if total_impact > 15:
            recommendations.append("🎯 **Cenário Altamente Favorável** - Implementação recomendada")
        elif total_impact > 5:
            recommendations.append("📈 **Cenário Positivo** - Considerar implementação")
        elif total_impact > -5:
            recommendations.append("⚖️ **Cenário Neutro** - Avaliar outros fatores")
        else:
            recommendations.append("⚠️ **Cenário Desfavorável** - Revisar ajustes propostos")
        
        # Recomendações baseadas em variáveis específicas
        high_impact_vars = [
            var for var, impact in individual_impacts.items() 
            if abs(impact['impact_percent']) > 10
        ]
        
        if high_impact_vars:
            recommendations.append(f"🔍 **Variáveis de Alto Impacto**: {', '.join(high_impact_vars)}")
        
        # Recomendações de sensibilidade
        high_sensitivity_vars = [
            var for var, sens in sensitivity_analysis.items() 
            if sens['sensitivity_range'] > 20
        ]
        
        if high_sensitivity_vars:
            recommendations.append(f"🎚️ **Variáveis Sensíveis**: {', '.join(high_sensitivity_vars)} - Monitorar cuidadosamente")
        
        # Recomendações gerais
        recommendations.extend([
            "📊 **Monitoramento**: Acompanhe resultados reais versus projetados",
            "🔄 **Ajustes Graduais**: Considere implementação faseada",
            "🎯 **Otimização Contínua**: Use dados reais para refinar o modelo"
        ])
        
        return recommendations

    def _assess_risk(self, individual_impacts, combined_impact):
        """Avalia riscos do cenário simulado"""
        risk_factors = []
        
        # Risco de volatilidade
        high_volatility_vars = [
            var for var, impact in individual_impacts.items() 
            if impact['volatility_impact'] > 15
        ]
        
        if high_volatility_vars:
            risk_factors.append(f"Alta volatilidade em: {', '.join(high_volatility_vars)}")
        
        # Risco de impacto negativo
        negative_impact_vars = [
            var for var, impact in individual_impacts.items() 
            if impact['impact_percent'] < -5
        ]
        
        if negative_impact_vars:
            risk_factors.append(f"Impacto negativo em: {', '.join(negative_impact_vars)}")
        
        # Risco geral
        total_impact = combined_impact.get('total_impact', 0)
        if total_impact < -10:
            risk_level = "Alto"
        elif total_impact < -5:
            risk_level = "Moderado"
        else:
            risk_level = "Baixo"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommended_actions': self._get_risk_mitigation_actions(risk_level, risk_factors)
        }

    def _get_risk_mitigation_actions(self, risk_level, risk_factors):
        """Retorna ações de mitigação de risco"""
        actions = []
        
        if risk_level == "Alto":
            actions.extend([
                "Revisar completamente os ajustes propostos",
                "Considerar cenários alternativos",
                "Implementar medidas de contingência"
            ])
        elif risk_level == "Moderado":
            actions.extend([
                "Monitorar indicadores-chave continuamente",
                "Preparar planos de ação para possíveis desvios",
                "Comunicar riscos às partes interessadas"
            ])
        else:
            actions.append("Manter monitoramento padrão")
        
        if risk_factors:
            actions.append("Endereçar fatores de risco específicos identificados")
        
        return actions
