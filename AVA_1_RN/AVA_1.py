# Análise dados de preço de casa
# Download da planilha via kaggle: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

scaler = StandardScaler()

# Importar planilha
arquivo = pd.read_csv('AVA_1_RN/Planilhas/diabetes_prediction_dataset.csv')

# Mostrar shape da planilha
#print(arquivo.shape)

# Apresentar informações sobre os dados da planilha
#print(arquivo.dtypes)

# Se necessarios excluir colunas

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

# Plotar o mapa de calor das correlações entre as variáveis
plt.figure(figsize=(10, 10))
sns.heatmap(arquivo.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor das Correlações entre Variáveis')
plt.show()

# Mostrar planilha
#print(arquivo.head())

# Definindo Variáveis de análise
y = arquivo['diabetes']
x = arquivo.drop('diabetes',axis=1)

# Imputar valores faltantes com a média da coluna
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Scalar valores
x = scaler.fit_transform(x)
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x,y, test_size=0.30)

# Treinar o modelo LogisticRegression
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(x_Treino, y_Treino)

# Fazer previsões e avaliar o modelo
resultado = modelo.score(x_Teste,y_Teste)
print("Resultado", resultado*100,"%")

predicao = modelo.predict(x_Teste)

matrix = confusion_matrix(y_Teste, predicao)
print(matrix)

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

# Resultados
print(f"Novo indivíduo: {novos_dados}")
print(f"Previsão de diabetes: {'Sim' if previsao[0] == 1 else 'Não'}")