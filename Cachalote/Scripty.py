import numpy as np
import pandas as pd
from keras import models, layers, optimizers

# Carregar o arquivo CSV com os dados do carro
arquivo = pd.read_csv('Cachalote/ToyotaCorolla.csv')

# Remover colunas irrelevantes e renomear conforme necessário
arquivo.rename(columns={'Mfg_Year': 'Ano',
                        'KM': 'Quilometragem',
                        'HP': 'Potencia',
                        'CC': 'Litro',
                        'Doors': 'Portas',
                        'Cylinders': 'Cilindrada',
                        'Price': 'Preco'}, inplace=True)

# Separar variáveis de entrada e saída
x = arquivo.drop(columns=['Model', 'Preco'])
y = arquivo['Model']

# Separar variáveis categóricas e numéricas
x_categorico = x[['Fuel_Type', 'Color', 'Automatic']]
x_numerico = x.drop(columns=['Fuel_Type', 'Color', 'Automatic'])

# Imputar dados faltantes
x_numerico = x_numerico.fillna(x_numerico.median())
x_categorico = x_categorico.fillna('missing')

# Codificar variáveis categóricas
x_categorico_encoded = pd.get_dummies(x_categorico)

# Combinar dados numéricos e categóricos
x_novo = np.hstack([x_numerico.values, x_categorico_encoded.values])

# Codificar a variável de saída
y_encoded = pd.get_dummies(y).values

# Dividir os dados em conjuntos de treino e teste
indices = np.arange(x_novo.shape[0])
np.random.shuffle(indices)
train_size = int(0.70 * len(indices))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

x_Treino = x_novo[train_indices].astype(np.float32)
x_Teste = x_novo[test_indices].astype(np.float32)
y_Treino = y_encoded[train_indices].astype(np.float32)
y_Teste = y_encoded[test_indices].astype(np.float32)

# Criar o modelo de rede neural
modelo = models.Sequential()
modelo.add(layers.Dense(128, input_dim=x_Treino.shape[1], activation='relu'))
modelo.add(layers.Dense(64, activation='relu'))
modelo.add(layers.Dense(32, activation='relu'))
modelo.add(layers.Dense(y_Treino.shape[1], activation='softmax'))

# Compilar o modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
historia = modelo.fit(x_Treino, y_Treino, epochs=200, batch_size=2, validation_data=(x_Teste, y_Teste), verbose=1)

# Avaliar o modelo
resultado = modelo.evaluate(x_Teste, y_Teste, verbose=0)
print("\n\nAcurácia")  
print("Resultado:", resultado[1] * 100, "%")
