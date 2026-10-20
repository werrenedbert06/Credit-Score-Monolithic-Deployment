import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

class CreditScorePreprocessor:
    """Menangani pembersihan data Credit Score, ekstraksi fitur, dan pembuatan transformer."""
    
    def __init__(self):
        self.test_size = 0.2
        self.random_state = 42

    def _convert_to_months(self, age_str):
        """Fungsi internal untuk mengubah format teks umur menjadi total bulan."""
        if pd.isna(age_str):
            return np.nan
        
        text = str(age_str).lower()
        years = re.findall(r'(\d+)\s*y', text)
        months = re.findall(r'(\d+)\s*m', text)

        y = int(years[0]) if years else 0
        m = int(months[0]) if months else 0
        return y * 12 + m
    
    def _cap_anomalies(self, df):
        """Capping anomali berdasarkan domain rules."""
        cap_rules = {'Age': (18, 100),
                    'Num_Bank_Accounts': (0, 20),
                    'Num_Credit_Card': (0, 20),
                    'Interest_Rate': (0, 100),
                    'Num_of_Loan': (0, 50),
                    'Delay_from_due_date': (0, None),
                    'Num_of_Delayed_Payment': (0, None),
                    'Changed_Credit_Limit': (None, None),
                    'Outstanding_Debt': (0, None),
                    'Credit_Utilization_Ratio': (None, 100),
                    'Num_Credit_Inquiries': (0, None),
                    'Total_EMI_per_month': (0, None),
                    'Amount_invested_monthly': (0, None),
        }

        for col, (lower, upper) in cap_rules.items():
            df[col] = df[col].clip(lower=lower, upper=upper)

        valid = df['Credit_History_Age_Months'].notna() & df['Age'].notna()
        df.loc[valid, 'Credit_History_Age_Months'] = np.minimum(
            df.loc[valid, 'Credit_History_Age_Months'],
            df.loc[valid, 'Age'] * 12)
        return df

    def clean_and_split(self, data_path: str | Path):
        """Memuat data, membersihkan anomali, melakukan mapping target, dan membagi train/test."""
        df = pd.read_csv(Path(data_path), sep=",")
        
        # 1. Drop Irrelevant Columns (Metadata)
        df = df.drop(columns=['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN'])

        # 2. Casting Feature where str to int/float
        CAST_COLS = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
                     'Outstanding_Debt', 'Changed_Credit_Limit', 'Amount_invested_monthly']
        for col in CAST_COLS:
            df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Parsing 'Credit_History_Age_Months'
        df['Credit_History_Age_Months'] = df['Credit_History_Age'].apply(self._convert_to_months)
        df.drop(columns=['Credit_History_Age'], inplace=True)

        # 4. Capping Anomalies
        df = self._cap_anomalies(df)

        # 5. Replace Inconsistent to NaN
        df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
        df['Type_of_Loan'] = df['Type_of_Loan'].astype(str).str.split(',').str[0].str.strip()
        df['Type_of_Loan'] = df['Type_of_Loan'].replace(['', 'nan', 'None'], np.nan)

        df_clean = df.reset_index(drop=True)

        # 6. Pemisahan Fitur dan Target (Mapping Ordinal)
        X = df_clean.drop(columns=['Credit_Score'])
        y = df_clean['Credit_Score'].map({'Poor': 0, 'Standard': 1, 'Good': 2})

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def get_transformer(self):
        """Merakit transformer numerik, ordinal, dan nominal sesuai rancangan notebook."""
        
        num_feat = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
                    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
                    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 
                    'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Months']
        
        ord_feat = ['Credit_Mix', 'Payment_of_Min_Amount']
        nom_feat = ['Month', 'Occupation', 'Type_of_Loan', 'Payment_Behaviour']

        num_preprocess = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        ord_preprocess = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('ord_encoder', OrdinalEncoder(categories=[
                ['Bad', 'Standard', 'Good'],
                ['No', 'Yes']
            ], handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        nom_preprocess = Pipeline([
            ('nom_imputer', SimpleImputer(strategy='most_frequent')),
            ('nom_encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        return ColumnTransformer(transformers=[
            ('num_preprocessor', num_preprocess, num_feat),
            ('ord_preprocessor', ord_preprocess, ord_feat),
            ('nom_preprocessor', nom_preprocess, nom_feat)
        ], remainder='drop')