'''
Porta AND:
x1 | x2 | y
0  | 0  | 0
0  | 1  | 0
1  | 0  | 0
1  | 1  | 1
'''

x1 = [0, 0, 1, 1] # Possíveis entradas de x1
x2 = [0, 1, 0, 1] # Possíveis entradas de x2
y = [0, 0, 0, 1] # Vetor para treinamento

w1 = 5.6
w2 = 11.7

bias = -0.5
tx = 0.4 # Coeficiente de inteligencia

threshold = 0
iteracao =  0 # Contagem de iteração

while True:
    certo = 0
    for i in range(4):
        h =  x1[i]*w1 + x2[i]*w2 + bias # Função hipotese

        # Função de ativação
        a = 1 if h > threshold else 0

        erro = tx*(y[i] - a)

        # Ajuste dos pesos da equação
        w1 += erro *x1[i]
        w2 += erro *x2[i]
        bias += erro

        print(f"bias = {bias:.2f} w1 = {w1:.2F} w2 = {w2:.2f}")

        if a == y[i]:
            print(f"Acertou: y[{i}]")
            certo += 1
        else:
            print(f"Errou: y[{i}]")

        iteracao+=1

    if certo == 4:
        break

print(f"Algoritmo treinado com {iteracao} iterações!")

# Teste do treinamento
print("Agora vamos testar o algoritmo! Porta AND!")

e1 = int(input("Digite a entrada 1: "))
e2 = int(input("Digite a entrada 1: "))

h = e1*w1 + e2*w2 + bias

a = 1 if h>threshold else 0

print(f"Saida AND: {a}")
