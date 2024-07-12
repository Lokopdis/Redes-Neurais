# Análise dados de preço de casa
# Download da planilha via kaggle: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Importar planilha
arquivo = pd.read_csv('Planilhas de Dados/wine_dataset.csv')

# Mostrar shape da planilha
print(arquivo.shape)

# Apresentar informações sobre os dados da planilha
print(arquivo.dtypes)

# Mostrar planilha
print(arquivo.head())

arquivo['style'] = arquivo['style'].apply(lambda x: 1 if x == 'red' else 0 if x == 'white' else x)

# Mostrar planilha atualizada
print(arquivo.head())

# Treinamento
y = arquivo['style']
x = arquivo.drop('style', axis=1)

x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x,y, test_size=0.3)

modelo=ExtraTreesClassifier()
modelo.fit(x_Treino,y_Treino)

# Testando
Resultado = modelo.score(x_Teste,y_Teste)
print("Resultado:", Resultado*100, "%")

arquivo['previsao'] = modelo.predict(x) 

print(arquivo.head())