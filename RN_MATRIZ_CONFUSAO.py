from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
x =pd.DataFrame(dados.data,columns=[dados.feature_names])#todas as outras colunas para entrada x
y = pd.Series(dados.target) # já definido o target noarquivo (tem ou não câncer)

# Análise de x
print(x.shape)

# Análise de y
print(y.shape)

print(y.value_counts())

# Treinamento
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x,y, test_size=0.30, random_state=9)

modelo = LogisticRegression(C=95)
modelo.fit(x_Treino,y_Treino)

# Teste
resultado = modelo.score(x_Teste, y_Teste)
print("Acurácia:",resultado*100,"%")

predicao = modelo.predict(x_Teste)

matrix = confusion_matrix(y_Teste, predicao)
print(matrix)