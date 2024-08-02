import warnings
warnings.filterwarnings('ignore')
import keras
from keras import datasets
from keras import utils
import matplotlib.pyplot as plt
from keras import models  #Sequential
from keras import layers #Dense, Activation
from keras import optimizers #SGD


#mnist é um conjunto de dados q possui 60 mil amostras de dígitos escritos a mão para treino,
#e 10 mil amostras de dígitos escritos a mão para teste, conforme abaixo.
(x_treino,y_treino),(x_teste,y_teste) = datasets.mnist.load_data()
y_treino #60 mil números de saída y de acordo com aentrada em imagem 
y_treino_convertido = utils.to_categorical(y_treino) #Convertendo a coluna de valores em uma matriz de classes
y_teste_convertido = utils.to_categorical(y_teste)

#plt.imshow(x_treino[0],cmap='gray')
#plt.show()

print(x_treino.shape)

#transforma o x de 28x28 em uma coluna de 784 píxel para entrada da RNA
x_treino_remodelado = x_treino.reshape((60000,784))
x_teste_remodelado = x_teste.reshape((10000,784))

#transforma o x de uma escala de 0 a 255 para umaescala de 0 a 1
x_treino_normalizado = x_treino_remodelado.astype('float32')/255
x_teste_normalizado = x_teste_remodelado.astype('float32')/255
x_treino_normalizado[0]

modelo = models.Sequential()
modelo.add(layers.Dense(30,input_dim=784,kernel_initializer='normal',activation='relu')) #Entrada e primeira camada de30 neurônios
modelo.add(layers.Dense(30, kernel_initializer='normal',activation='relu')) #Camada intermediária de 30neurônios
modelo.add(layers.Dense(10, kernel_initializer='normal',activation='softmax'))#Saída 10 neurônios. 10 números

print("1")

otimizador = optimizers.SGD()
modelo.compile(loss='categorical_crossentropy', optimizer = otimizador, metrics=['acc']) #'acc' é amétrica de acurácia
historico = modelo.fit(x_treino_normalizado,y_treino_convertido,epochs=40,batch_size=100,validation_data=(x_teste_normalizado,y_teste_convertido),verbose=1)

#relação da acurácia de treino e teste
historico.history['acc']
historico.history['val_acc']

acuracia_treino = historico.history['acc']
acuracia_teste =historico.history['val_acc']
epochs = range(1,len(acuracia_treino)+1)
plt.plot(epochs,acuracia_treino, '-g', label = 'Acurácia dados de treino')
plt.plot(epochs,acuracia_teste, '-b', label = 'Acurácia dados de teste')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.show()