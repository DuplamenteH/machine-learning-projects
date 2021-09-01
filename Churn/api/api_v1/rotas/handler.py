

from flask import Flask,request
import pickle
import pandas as pd
from parameter.processing.data_prep import Data_Prepared
#from parameter.data_prep import Data_Prepared



#load model
model = pickle.load(open('../../model/churn_randomFlorest.pkl','rb'))
#/media/cmatheus/dadosProjetos/portfolio/Churn/api/api_v1/model/churn_randomFlorest.pkl


#instanciando o flask

app = Flask(__name__)



#criando endpoints;
@app.route('/',methods=['GET'])
def return_main_page():
    return "<p><h1>Pagina inicial da api<h1></p>"


@app.route('/predict',methods=['POST'])
def predict():
    dados_json=request.get_json()
    #coletando os dados
    if dados_json:
        if isinstance(dados_json,dict):
            df_raw = pd.DataFrame(dados_json,index=[0])
        else:
            df_raw = pd.DataFrame(dados_json,columns=dados_json[0].keys() )

    #predict
    df_raw = Data_Prepared.get_df_transform(df_raw)
    pred = model.predict(df_raw)

    df_raw['predicao_churn'] = pred

    return df_raw.to_json(orient='records')



if __name__=='__main__':
    #start flask
    app.run(host='0.0.0.0', port='3333')