# modules/scenario_simulator.py
import plotly.graph_objects as go

class ScenarioSimulator:
    def simulate(self, data, scenario_variables, adjustments: dict, ai_core=None):
        try:
            impacts = {}
            for var, pct in adjustments.items():
                if var in data.columns:
                    mean = data[var].mean()
                    new_mean = mean * (1 + pct)
                    impact_pct = ((new_mean - mean) / (mean + 1e-9)) * 100
                    impacts[var] = float(impact_pct)
            # build bar plot
            fig = go.Figure()
            vars = list(impacts.keys())
            vals = [impacts[v] for v in vars]
            colors = ['green' if v>=0 else 'red' for v in vals]
            fig.add_trace(go.Bar(x=vars, y=vals, marker_color=colors))
            fig.update_layout(title="Impacto percentual estimado por variável", yaxis_title="Impacto (%)")
            total_impact = (sum(abs(v) for v in vals)/len(vals)) if vals else 0.0
            recs = ["Monitorar após implementar", "Validar com amostra A/B"]
            return {'variable_impacts': impacts, 'comparison_plot': fig, 'total_impact': total_impact, 'recommendations': recs}
        except Exception as e:
            return {'error': str(e)}
