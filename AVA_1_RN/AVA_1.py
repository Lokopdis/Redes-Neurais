# Análise dados de preço de casa
# Download da planilha via kaggle: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Importar planilha
arquivo = pd.read_csv('AVA_1_RN/Planilhas/diabetes_prediction_dataset.csv')

# Mostrar shape da planilha
#print(arquivo.shape)

# Apresentar informações sobre os dados da planilha
#print(arquivo.dtypes)

# Mapear valores de 'gender' para números
arquivo['gender'] = arquivo['gender'].map({'Male': 0, 'Female': 1})

arquivo['smoking_history'] = arquivo['smoking_history'].map({
    'never': 0,
    'No Info': 1,
    'former': 2,
    'current': 3,
    'ever': 4,
    'not current': 5
})


# Mostrar primeiras linhas para confirmar mudanças
#print(arquivo.head())

# Mostrar planilha
#print(arquivo.head())

# Definindo Variáveis de análise
y = arquivo['diabetes']
x = arquivo.drop('diabetes',axis=1)

# Imputar valores faltantes com a média da coluna
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)
x = scaler.fit_transform(x)
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x,y, test_size=0.30)

# Treinar o modelo LogisticRegression
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(x_Treino, y_Treino)

# Fazer previsões e avaliar o modelo
resultado = modelo.score(x_Teste,y_Teste)
print("Resultado", resultado*100,"%")

# Gerar novos dados aleatórios
novos_dados = {
    'gender': np.random.choice([0, 1]),
    'age': np.random.uniform(18, 100),
    'hypertension': np.random.choice([0, 1]),
    'heart_disease': np.random.choice([0, 1]),
    'smoking_history': np.random.choice([0, 1, 2, 3, 4, 5]),
    'bmi': np.random.uniform(15, 50),
    'HbA1c_level': np.random.uniform(3, 15),
    'blood_glucose_level': np.random.uniform(50, 300)
}

# Converter o indivíduo para DataFrame
novos_dados_df = pd.DataFrame([novos_dados])


# Imputar valores faltantes e escalar os novos dados
novos_dados_processed = scaler.transform(imputer.transform(novos_dados_df))

# Fazer previsão para os novos dados
previsao = modelo.predict(novos_dados_processed)
proba = modelo.predict_proba(novos_dados_processed)

# Resultados
print(f"Novo indivíduo: {novos_dados}")
print(f"Previsão de diabetes: {'Sim' if previsao[0] == 1 else 'Não'}")