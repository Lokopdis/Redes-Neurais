import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

# Carregar o dataset
arquivo = pd.read_csv('Testes AVA/Dados/diabetes_prediction_dataset.csv')

# Analisar os dados presentes na tabela
# Mostrar shape da planilha
print(arquivo.shape)

# Apresentar informações sobre os dados da planilha
print(arquivo.dtypes)

# Mostrar planilha
print(arquivo.head())

# Não há necessidade de remover colunas da tabela

# Verificar as colunas categóricas
print(arquivo['gender'].unique())
print(arquivo['smoking_history'].unique())


X = arquivo.drop(columns=['diabetes'])
y = arquivo['diabetes']

# Identificar colunas categóricas e numéricas
categorical_features = ['gender', 'smoking_history']
numeric_features = X.columns.difference(categorical_features)

# Criar transformers para colunas categóricas e numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar transformers em um ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print(arquivo.head())

# Criar pipeline com preprocessor e modelo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Dividir o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred)*100,"%")
print(classification_report(y_test, y_pred))

# Gerar novos dados aleatórios
novos_dados = {
    'gender': np.random.choice(['Male', 'Female']),
    'age': np.random.uniform(18, 100),
    'hypertension': np.random.choice([0, 1]),
    'heart_disease': np.random.choice([0, 1]),
    'smoking_history': np.random.choice(['never', 'No Info', 'former', 'current', 'ever', 'not current']),
    'bmi': np.random.uniform(15, 50),
    'HbA1c_level': np.random.uniform(3, 15),
    'blood_glucose_level': np.random.uniform(50, 300)
}

# Converter o indivíduo para DataFrame
novos_dados_df = pd.DataFrame([novos_dados])

# Fazer previsão para os novos dados
previsao = pipeline.predict(novos_dados_df)

# Resultados
print(f"Novo indivíduo: {novos_dados}")
print(f"Previsão de diabetes: {'Sim'if previsao[0] == 1 else 'Não'}")
