from sklearn import preprocessing
import pandas as pd
import unidecode
import numpy as np

class run_out():

    def __init__(self, base, param1=None, param2=None, param3=None, param4=None):

        self.base = base

    def execute(self):

        
        cols_remov_carc = ['dsc_status_contrato', 'dsc_tecnologia_dados', 'grupo2_rfv', 'grupo2_rfv_ant', 'flag_meutim', 'fx_score_credito', 'nm_regiao']
        for c in cols_remov_carc:
            self.base[c] = self.base[c].apply(lambda x: str(x).replace(' / ', '_'))

        self.base['flag_meutim'] = self.base['flag_meutim'].apply(lambda x: str(x).replace(' + ', ' e '))

        for c in cols_remov_carc:
            self.base[c] = self.base[c].apply(lambda x: str(x).replace('/', '_'))

        for c in cols_remov_carc:
            self.base[c] = self.base[c].apply(lambda x: str(x).replace(' ', '_'))

        for c in cols_remov_carc:
            self.base[c] = self.base[c].apply(lambda x: str(x).replace('-', '_'))
            
        for c in cols_remov_carc:
            self.base[c] = self.base[c].apply(lambda x: unidecode.unidecode(str(x)))
        
        
        for col in ['dsc_status_contrato', 'dsc_tecnologia_dados', 'grupo2_rfv', 'grupo2_rfv_ant', 'flag_meutim', 'fx_score_credito', 'nm_regiao']:
            
            if col == 'dsc_status_contrato':
                values = ['ATIVO','BLOQUEADO','CONVERTIDO','DESATIVADO','INATIVO','SUSPENSO']
            elif col == 'dsc_tecnologia_dados':
                values = ['2G','3G','4G','Sem_uso']
            elif col == 'grupo2_rfv':
                values = ['Esporadico','Fiel_Alto_Valor_Recente','Fiel_Baixo_Med_Valor_Recente','Sazonal_Baixa_Recencia','Sazonal_Baixo_Alto_Valor_Recente','Sem_Recarga']
            elif col == 'grupo2_rfv_ant':
                values = ['Esporadico','Fiel_Alto_Valor_Recente','Fiel_Baixo_Med_Valor_Recente','Sazonal_Baixa_Recencia','Sazonal_Baixo_Alto_Valor_Recente','Sem_Recarga']
            elif col == 'flag_meutim':
                values = ['App','App_e_Meu_Plano','App_e_Web','Meu_Plano','Sem_Uso','Todos','Web','Web_e_Meu_Plano']
            elif col == 'fx_score_credito':
                values = ['01_Very_High_Risk','02_High_Risk','03_Intermediate_Risk','04_Intermediate_Risk','04_Low_Risk','05__Very_Low_Risk','06_No_Information']
            elif col == 'nm_regiao':
                values = ['Centro_oeste','NULL','Nordeste','Norte','Sudeste','Sul']

            for var in values:
                self.base[f'{col}_{var}'] = self.base[col].apply(lambda x: 1 if x == var else 0)
            
            