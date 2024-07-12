# Análise dados de preço de casa
# Download da planilha via kaggle: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importar planilha
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv('Planilhas de Dados/kc_house_data.csv')

# Mostrar shape da planilha
print(arquivo.shape)

# Apresentar informações sobre os dados da planilha
print(arquivo.dtypes)

# Mostrar planilha
print(arquivo.head())

# Excluir dados irrelevantes da tabela
arquivo.drop('id', axis=1,inplace=True)
arquivo.drop('date', axis=1,inplace=True)
arquivo.drop('zipcode', axis=1,inplace=True)
arquivo.drop('lat', axis=1,inplace=True)
arquivo.drop('long', axis=1,inplace=True)

# Mostrar planilha atualizada
print(arquivo.head())

# Definindo Variáveis de análise
y = arquivo['price']
x = arquivo.drop('price',axis=1)

# Treinando a Rede
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x,y, test_size=0.30)

modelo = LinearRegression()
modelo.fit(x_Treino,y_Treino)

# Etapa de teste
resultado = modelo.score(x_Teste,y_Teste)
print("Resultado:",resultado*100, "%" )