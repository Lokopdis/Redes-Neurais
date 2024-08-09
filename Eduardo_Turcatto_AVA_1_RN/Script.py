# BIBLIOTÉCAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Carregar DataSet
Dados = pd.read_csv('Eduardo_Turcatto_AVA_1_RN/Dados/diabetes_prediction_dataset.csv')

# Analisar dados presentes na Tabela
# Mostrar shape da planilha
print(Dados.shape)

# Apresentar informações sobre os dados da planilha
print(Dados.dtypes)

# Mostrar planilha
print(Dados.head())

# Não há necessidade de remover nenhum item da tabela
# Em caso de necessidade de remoção de colunas usar o comando:
# DATA_SET.drop()

# Traduzir título das colunas para facilitar leitura
# Criar um dicionário para renomear as colunas com tradução para o português
Novas_Colunas = {
    'gender': 'Gênero',
    'age': 'Idade',
    'hypertension': 'Hipertensão',
    'heart_disease': 'Doença_Cardiaca',
    'smoking_history': 'Histórico_de_Fumo',
    'bmi': 'IMC',
    'HbA1c_level': 'Nível_de_HbA1c',
    'blood_glucose_level': 'Nível_de_Glicose',
    'diabetes': 'Diagnóstico'  # Correção do nome da coluna 'diabetes'
}

Dados.rename(columns=Novas_Colunas, inplace=True)

# Definindo variáveis do X e Y para o sistema
x = Dados.drop(columns=['Diagnóstico'])
y = Dados['Diagnóstico']

##############################################################

# Matriz de correlação
# É preciso garantir apenas dados numéricos
# Selecionar apenas colunas numéricas
x_numeric = x.select_dtypes(include=[np.number])

# Calcular a matriz de correlação
corr_matrix = x_numeric.corr()
print(corr_matrix)

# Plotar a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

###############################################################

# Definindo etapas de pré-processamento para treinamento da Rede Neural
# Verificar as colunas categóricas
print(Dados['Gênero'].unique())
print(Dados['Histórico_de_Fumo'].unique())

# Separar colunas categóricas e numéricas
Categorical_Features = ['Gênero', 'Histórico_de_Fumo']
Numeric_Features = x.columns.difference(Categorical_Features)

# Criar etapas de transformações das colunas categóricas e numéricas
Transformacao_Numerica = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

Transformacao_Categorica = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Correção do parâmetro handle_unknown
])

# Criar etapa de processamento para transformar colunas com ColumnTransformer
Processar = ColumnTransformer(
    transformers=[
        ('numerico', Transformacao_Numerica, Numeric_Features),
        ('categorico', Transformacao_Categorica, Categorical_Features)
    ]
)

# Criar etapa de operação do modelo
pipeline = Pipeline(steps=[
    ('processo', Processar),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Dividir dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)

# Realizar treinamento do modelo
pipeline.fit(x_treino, y_treino)

# Previsão para verificar acurácia do modelo
y_pred = pipeline.predict(x_teste)
print("Acurácia:", accuracy_score(y_teste, y_pred) * 100, "%")

# Gerar novos dados aleatórios
# Introduzir elemento aleatório para teste
novos_dados = {
    'Gênero': np.random.choice(['Male', 'Female']),
    'Idade': np.random.uniform(18, 100),
    'Hipertensão': np.random.choice([0, 1]),
    'Doença_Cardiaca': np.random.choice([0, 1]),
    'Histórico_de_Fumo': np.random.choice(['never', 'No Info', 'former', 'current', 'ever', 'not current']),
    'IMC': np.random.uniform(15, 50),
    'Nível_de_HbA1c': np.random.uniform(3, 15),
    'Nível_de_Glicose': np.random.uniform(50, 300)
}

# Converter o indivíduo para DataFrame
novos_dados_df = pd.DataFrame([novos_dados])

# Fazer previsão para os novos dados
previsao = pipeline.predict(novos_dados_df)

# Resultados
print(f"Novo indivíduo: {novos_dados}")
print(f"Previsão de diabetes: {'Sim' if previsao[0] == 1 else 'Não'}")