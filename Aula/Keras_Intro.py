import warnings
from sklearn.datasets import load_iris
import pandas as pd
from keras import utils
from sklearn.model_selection import train_test_split

from keras import models
from keras import layers
from keras import optimizers

warnings.filterwarnings('ignore')

# Carregar o dataset Iris
iris = load_iris()

# Criar DataFrame com os dados
x = pd.DataFrame(iris.data, columns=iris.feature_names)

# Criar Series com os alvos
y = pd.Series(iris.target)

# Converter os alvos para one-hot encoding
y_convertido = utils.to_categorical(y)

# Mostrar algumas informações para verificar
print("Primeiras 5 linhas de x:")
print(x.head())
print("\nPrimeiras 5 linhas de y_convertido:")
print(y_convertido[:5])

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y_convertido, test_size = 0.3)

modelo = models.Sequential() #RNA conectada adiante
#Construimos a rede neural adicionando camadas por meio de comando".add". O primeiro argumento consiste no número de neurônios
#que aquela camada terá. Também podemos informar o modelo de inicialização das variváveis e a função de ativação nessa camada.
modelo.add(layers.Dense(10,input_dim=4,kernel_initializer='normal',activation='relu'))# camada oculta terá 10 neurônios, input_dim é
#a quantidade de neurônios de entrada, que nesse caso são 4 variáveispreditoras
modelo.add(layers.Dense(3, kernel_initializer='normal', activation='softmax'))
#Essa é a saída (3 neurônios pos são 3 classes)
#A função softmax para a saída costuma ser usada em problemas de classificação. Ela dá como resultado a probabilidade de pertencer a cada uma das classes

otimizador = optimizers.SGD()
#loss custo para a resposta (próx. 0 ideal).
modelo.compile(loss='categorical_crossentropy', optimizer = otimizador, metrics=['acc'])#'acc'é a métrica de acurácia
modelo.fit(x_treino,y_treino,
epochs=1000,batch_size=105,validation_data=(x_teste,y_teste),verbose=1)