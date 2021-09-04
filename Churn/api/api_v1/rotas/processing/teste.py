import pandas as pd
from data_prep import Data_Prepared

dados = pd.read_csv('/media/cmatheus/dadosProjetos/portfolio/Churn/dados/customer-churn-prediction-2020/train.csv')

dp = Data_Prepared()

print(dados.info())


dados_tratados = dp.get_df_transform(dados)
print(dados_tratados.info())