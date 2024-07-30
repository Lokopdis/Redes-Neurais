import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Carregar o arquivo CSV
arquivo = pd.read_csv('Planilhas de Dados/ford.csv')

# Listar valores únicos das colunas 'model' e 'transmission'
unique_models = arquivo['model'].unique()
unique_transmissions = arquivo['transmission'].unique()

# Criar mapeamentos
model_mapping = {model: idx for idx, model in enumerate(unique_models)}
transmission_mapping = {trans: idx for idx, trans in enumerate(unique_transmissions)}

# Aplicar mapeamentos às colunas
arquivo['model'] = arquivo['model'].map(model_mapping)
arquivo['transmission'] = arquivo['transmission'].map(transmission_mapping)

# Codificar a coluna 'fuelType'
arquivo['fuelType'] = arquivo['fuelType'].map({'Petrol': 0, 'Diesel': 1})
print(arquivo.dtypes)

print("Model Mapping: ", model_mapping)
print("Transmission Mapping: ", transmission_mapping)

# Definindo variáveis de análise
y = arquivo['price']
x = arquivo.drop('price', axis=1)

# Imputar valores faltantes com a média da coluna
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Escalar valores
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Dividir os dados em conjuntos de treino e teste
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x, y, test_size=0.30, random_state=42)

# Treinar o modelo LinearRegression
modelo = LinearRegression()
modelo.fit(x_Treino, y_Treino)

# Fazer previsões e avaliar o modelo
resultado = modelo.score(x_Teste, y_Teste)
print("Resultado:", resultado * 100, "%")

# Gerar novos dados aleatórios
novos_dados = {
    'model': np.random.randint(0, 23),
    'year': np.random.randint(1990, 2023),
    'transmission': np.random.randint(0, 2),
    'mileage': np.random.randint(0, 200000),
    'fuelType': np.random.choice([0,1]),
    'tax': np.random.uniform(0, 600),
    'mpg': np.random.uniform(0, 100),
    'engineSize': np.random.uniform(0.5, 6.0)
}

# Converter o novo indivíduo em um DataFrame
novo_df = pd.DataFrame([novos_dados])

# Converter o DataFrame para um array NumPy antes de escalar
novo_df_np = novo_df.to_numpy()

# Aplicar a transformação de escala usando o scaler treinado
novo_df_scaled = scaler.transform(novo_df_np)

# Fazer a previsão
previsao = modelo.predict(novo_df_scaled)

print("Novo indivíduo:")
print(novo_df)
print("\nPrevisão de preço esta entre: $", previsao[0]*resultado,"e: $", previsao[0]*(1+(1-resultado)), "\nCom a média de preço em: $", previsao[0])
