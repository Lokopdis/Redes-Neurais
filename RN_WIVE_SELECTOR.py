# Análise dados de preço de casa
# Download da planilha via kaggle: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Importar planilha
pd.set_option('display.max_columns', 21)
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
