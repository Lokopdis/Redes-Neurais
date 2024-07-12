# Geração de dados pseudo aleatórios

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Gerando dados
x, y = make_regression(n_samples = 200, n_features = 1, noise = 30)

# Definir o algoritmo que sera utilizado
modelo = LinearRegression() 

# Executando o modelo
modelo.fit(x,y)

# Coeficiente linear b
b = modelo.intercept_
print(b)

# Coeficiente angular a
a = modelo.coef_
print(a)

# Fase de testes
# Treino 70% | Test 30%
x_Treino, x_Teste, y_Treino, y_Teste = train_test_split(x,y,test_size = 0.30)

# Treinando a Rede
modelo.fit(x_Treino, y_Treino)

# Testando a Rede
resultado = modelo.score(x_Teste,y_Teste)
print("Resultado: ")
print(resultado*100)

# Plotando os dados
plt.scatter(x, y)
plt.show()

''' 
# Mostrar dados gerados
# Plotando os dados
plt.scatter(x, y)
plt.show()

# Mostra dadods com reta
# Plotando reta resultante
plt.scatter(x,y)
xreg = np.arange(-3,3,1)
plt.plot(xreg, a*xreg + b, color = 'red') # Reta ax+b
plt.show()
'''