import pandas as pd
import numpy as np
import io

class DataLoader:
    """Classe para carregamento de dados"""
    
    def load_data(self, uploaded_file):
        """Carrega dados de arquivo upload"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'csv':
                return pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                return pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                return pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Formato não suportado: {file_extension}")
        except Exception as e:
            raise Exception(f"Erro ao carregar arquivo: {str(e)}")
    
    def validate_data(self, data):
        """Valida a estrutura dos dados"""
        if data.empty:
            raise ValueError("Dados vazios")
        
        # Verificar se há dados numéricos para análise
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Nenhuma coluna numérica encontrada para análise")
        
        return True