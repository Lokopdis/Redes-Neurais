import pandas as pd
import numpy as np
from keras import utils
from keras import models
from keras import layers
from keras import optimizers
import warnings

warnings.filterwarnings('ignore')

# Carregar o dataset
Dados = pd.read_csv('Eduardo_Turcatto_AVA_1_RN/Dados/diabetes_prediction_dataset.csv')

# Traduzir título das colunas para facilitar leitura
Novas_Colunas = {
    'gender': 'Gênero',
    'age': 'Idade',
    'hypertension': 'Hipertensão',
    'heart_disease': 'Doença_Cardiaca',
    'smoking_history': 'Histórico_de_Fumo',
    'bmi': 'IMC',
    'HbA1c_level': 'Nível_de_HbA1c',
    'blood_glucose_level': 'Nível_de_Glicose',
    'diabetes': 'Diagnóstico'
}
Dados.rename(columns=Novas_Colunas, inplace=True)

# Codificar variáveis categóricas manualmente
Dados['Gênero'] = Dados['Gênero'].map({'Male': 1, 'Female': 0})
# Aplicar One-Hot Encoding para 'Histórico_de_Fumo'
Dados = pd.get_dummies(Dados, columns=['Histórico_de_Fumo'])

# Definir variáveis de entrada e saída
x = Dados.drop(columns=['Diagnóstico']).values
y = Dados['Diagnóstico'].values

# Normalizar os dados manualmente (min-max normalization)
x_min = np.min(x, axis=0)
x_max = np.max(x, axis=0)
x_normalizado = (x - x_min) / (x_max - x_min)

# Converter os dados para o tipo float32
x_normalizado = x_normalizado.astype(np.float32)

# Converter os alvos para formato categórico (one-hot encoding)
y_convertido = utils.to_categorical(y, 2).astype(np.float32)

# Dividir os dados manualmente em treino e teste
indices = np.arange(x.shape[0])
np.random.shuffle(indices)

train_size = int(0.7 * len(indices))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

x_treino = x_normalizado[train_indices]
y_treino = y_convertido[train_indices]
x_teste = x_normalizado[test_indices]
y_teste = y_convertido[test_indices]

# Criar o modelo de rede neural sequencial
modelo = models.Sequential()

# Adicionar a camada oculta com 10 neurônios e função de ativação ReLU
modelo.add(layers.Dense(10, input_dim=x_treino.shape[1], kernel_initializer='normal', activation='relu'))

# Adicionar a camada de saída com 2 neurônios e função de ativação softmax
modelo.add(layers.Dense(2, kernel_initializer='normal', activation='softmax'))

# Compilar o modelo com otimizador SGD, função de perda categorical_crossentropy, e a métrica de acurácia
otimizador = optimizers.SGD()
modelo.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc'])

# Treinar o modelo
modelo.fit(x_treino, y_treino, epochs=20, batch_size=105, validation_data=(x_teste, y_teste), verbose=1)

# Avaliar o modelo
resultado = modelo.evaluate(x_teste, y_teste, verbose=0)
print(f"\nAcurácia no conjunto de teste: {resultado[1] * 100:.2f}%")

# Fazer previsão para novos dados
novos_dados = {
    'Gênero': np.random.choice([1, 0]),
    'Idade': np.random.uniform(18, 100),
    'Hipertensão': np.random.choice([0, 1]),
    'Doença_Cardiaca': np.random.choice([0, 1]),
    'Histórico_de_Fumo_0': 0,
    'Histórico_de_Fumo_1': 0,
    'Histórico_de_Fumo_2': 0,
    'Histórico_de_Fumo_3': 0,
    'Histórico_de_Fumo_4': 0,
    'Histórico_de_Fumo_5': 0,
    'IMC': np.random.uniform(15, 50),
    'Nível_de_HbA1c': np.random.uniform(3, 15),
    'Nível_de_Glicose': np.random.uniform(50, 300)
}

# Atualizar o valor da categoria de fumo
novo_hist_fumo = np.random.choice([0, 1, 2, 3, 4, 5])
novos_dados[f'Histórico_de_Fumo_{novo_hist_fumo}'] = 1

# Converter o indivíduo para array e normalizar
novos_dados_array = np.array(list(novos_dados.values())).reshape(1, -1).astype(np.float32)
novos_dados_normalizado = (novos_dados_array - x_min) / (x_max - x_min)

# Certificar-se de que os dados normalizados estão no formato float32
novos_dados_normalizado = novos_dados_normalizado.astype(np.float32)

# Verifique o tipo e o conteúdo dos novos dados normalizados
print(f"Tipo dos novos dados normalizados: {novos_dados_normalizado.dtype}")
print(f"Forma dos novos dados normalizados: {novos_dados_normalizado.shape}")

# Fazer previsão
previsao = modelo.predict(novos_dados_normalizado)
print(f"Novo indivíduo: {novos_dados}")
print(f"Previsão de diabetes: {'Sim' if np.argmax(previsao) == 1 else 'Não'}")
