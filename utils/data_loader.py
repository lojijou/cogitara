# modules/data_loader.py
import pandas as pd

class DataLoader:
    def load_data(self, uploaded_file):
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith((".xls",".xlsx")):
            return pd.read_excel(uploaded_file)
        elif name.endswith(".json"):
            return pd.read_json(uploaded_file)
        else:
            raise ValueError("Formato não suportado. Use CSV, XLSX ou JSON.")

    def validate_has_numeric(self, df):
        return len(df.select_dtypes(include=['number']).columns) > 0
