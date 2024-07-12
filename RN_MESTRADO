import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importar planilha
arquivo = pd.read_csv('Planilhas de Dados/Admission_Predict_Ver1.1.csv')

# Definindo Variáveis de análise
x = arquivo.drop(['Serial No.', 'Chance of Admit '], axis=1)
y = arquivo['Chance of Admit ']

# Treinando o Modelo
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x, y, test_size=0.30, random_state=42)

modelo = LinearRegression()
modelo.fit(x_Treino, y_Treino)

# Avaliando o Modelo
resultado = modelo.score(x_Teste, y_Teste)
print("Resultado do modelo:", resultado * 100, "%")

# Exemplo de predição com novos dados
novos_dados = pd.DataFrame({
    'GRE Score': [280],
    'TOEFL Score': [110],
    'University Rating': [4],
    'SOP': [2.5],
    'LOR ': [2.5],  # Certifique-se de que todas as colunas usadas no treinamento estão presentes aqui
    'CGPA': [8.75],
    'Research': [1]
})

# Fazer previsão
previsao = modelo.predict(novos_dados[x.columns])  # Usar apenas as colunas usadas no treinamento
print("\nPrevisão de Chance de Admissão:", previsao*100,"%")
