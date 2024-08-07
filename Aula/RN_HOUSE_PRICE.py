# Análise dados de preço de casa
# Download da planilha via kaggle: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Importar planilha
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv('Planilhas de Dados/kc_house_data.csv')

# Mostrar shape da planilha
print(arquivo.shape)

# Apresentar informações sobre os dados da planilha
print(arquivo.dtypes)

# Mostrar planilha
print(arquivo.head())

'''
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

# Definir o dicionário de mapeamento de números para nomes
number_to_name = {
    0: "Alice",
    1: "Bob",
    2: "Charlie",
    3: "David",
    4: "Eve",
    5: "Frank",
    6: "Grace",
    7: "Hank",
    8: "Ivy",
    9: "Jack",
    10: "Kathy",
    11: "Leo",
    12: "Mia",
    13: "Nina",
    14: "Oscar",
    15: "Paul",
    16: "Quincy",
    17: "Rachel",
    18: "Steve",
    19: "Tina",
    20: "Uma",
    21: "Victor",
    22: "Wendy",
    23: "Xander"
}

def get_name_from_number(number):
    # Usar o dicionário para obter o nome correspondente
    return number_to_name.get(number, "Nome não encontrado")

# Testando a função com todos os valores de 0 a 23
for i in range(24):
    print(f"{i}: {get_name_from_number(i)}")
'''