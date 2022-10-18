from sklearn import preprocessing
import pandas as pd
import unidecode

class run_out():

    def __init__(self, base, param1=None, param2=None, param3=None, param4=None):

        self.base = base
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4

    def execute(self):
        
        # Filtrando
        self.base = self.base[self.base["dsc_plat_prod_contratado_m1"] == "PRE"]
        
        # Removendo variáveis
        self.base.drop(['id_user','ddd', 'dsc_plat_prod_contratado', 'dsc_plat_prod_contratado_m1', 'dsc_tipo_familia_plano', 'sum_vlr_fatura', 'atraso_ult_3M'],axis=1, inplace=True)
        
        # Filtrando
        self.base = self.base[self.base["nivel_plano_detalhado"] != "FIXO"]

        # Data imputation nas colunas numericas
        for col in ['tot_volume_dados', 'tot_volume_voz']:
            self.base[col].fillna(self.base[col].mean(), inplace=True)
            
        # Data imputation nas colunas categoricas
        for col in ['grupo2_rfv_ant', 'fx_uso_dias_RedeSocial', 'fx_uso_dias_Facebook', 'fx_uso_dias_Instagram', 'fx_uso_dias_LinkedIn', 'fx_uso_dias_Pinterest', 'fx_uso_dias_SnapChat', 'fx_uso_dias_TikTok', 'fx_uso_dias_Twitter']:
            self.base[col].fillna(self.base[col].value_counts().idxmax(), inplace=True)

        # Limpando caracteres especiais
        df_cols = pd.DataFrame(self.base.columns, columns=['cols'])
        df_cols['cols'] = df_cols['cols'].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True)
        self.base.columns = df_cols['cols'].values
        for c in self.base.select_dtypes(include=['object','category']).columns:
            self.base[c] = self.base[c].replace('/', '_', regex=True).replace('\W+', '', regex=True).replace('__', '', regex=True).str.lower()
        for c in self.base.select_dtypes(include=['object','category']).columns:
            self.base[c] = self.base[c].apply(lambda x: unidecode.unidecode(str(x)))
            
        # Transformando em colunas dummies
        dummies = pd.get_dummies(self.base[['dsc_status_contrato', 'nm_cluster', 'grupo2_rfv', 'flg_em_dia', 'nivel_plano_detalhado', 'fx_score_credito', 'sub_segmento_plano', 'nm_sit', 'nm_sit_weekend', 'dsc_plano_tarifario', 'grupo2_rfv_ant']], drop_first=False)
        self.base = pd.concat([self.base.drop(['dsc_status_contrato', 'nm_cluster', 'grupo2_rfv', 'flg_em_dia', 'nivel_plano_detalhado', 'fx_score_credito', 'sub_segmento_plano', 'nm_sit', 'nm_sit_weekend', 'dsc_plano_tarifario', 'grupo2_rfv_ant'],axis=1), dummies], axis=1)

        # Transformando em colunas labelEncoder
        cols = self.base.columns.tolist()
        cols_fx_uso_dias = []
        for col in cols:
            if col[0:11] == 'fx_uso_dias':
                cols_fx_uso_dias.append(col)
        for col in cols_fx_uso_dias:
            self.base[col] = self.base[col].replace(['4nonavigation','3lowuser','2miduser','1heavyuser'], [0,1,2,3])
    