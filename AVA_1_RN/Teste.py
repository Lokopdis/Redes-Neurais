import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Carregar o dataset
arquivo = pd.read_csv('AVA_1_RN/Planilhas/diabetes_prediction_dataset.csv')

# Verificar as colunas categóricas
print(arquivo['gender'].unique())
print(arquivo['smoking_history'].unique())

# Separar as features (X) e o target (y)
X = arquivo.drop(columns=['diabetes'])
y = arquivo['diabetes']

# Identificar colunas categóricas e numéricas
categorical_features = ['gender', 'smoking_history']
numeric_features = X.columns.difference(categorical_features)

# Imputação de valores faltantes para colunas numéricas
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

# Escalonamento das colunas numéricas
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Imputação de valores faltantes para colunas categóricas
categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# One-Hot Encoding para colunas categóricas
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_categorical = encoder.fit_transform(X[categorical_features])
encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

# Concatenar colunas numéricas e categóricas codificadas
X_processed = pd.concat([X[numeric_features].reset_index(drop=True), encoded_categorical_df.reset_index(drop=True)], axis=1)

# Dividir o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Treinar o modelo LogisticRegression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
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

# Imputar e escalar os novos dados
novos_dados_df[numeric_features] = numeric_imputer.transform(novos_dados_df[numeric_features])
novos_dados_df[numeric_features] = scaler.transform(novos_dados_df[numeric_features])
novos_dados_df[categorical_features] = categorical_imputer.transform(novos_dados_df[categorical_features])
encoded_novos_dados = encoder.transform(novos_dados_df[categorical_features])
encoded_novos_dados_df = pd.DataFrame(encoded_novos_dados, columns=encoder.get_feature_names_out(categorical_features))
novos_dados_processed = pd.concat([novos_dados_df[numeric_features].reset_index(drop=True), encoded_novos_dados_df.reset_index(drop=True)], axis=1)

# Fazer previsão para os novos dados
previsao = logreg.predict(novos_dados_processed)

# Resultados
print(f"Novo indivíduo: {novos_dados}")
print(f"Previsão de diabetes: {'Sim' if previsao[0] == 1 else 'Não'}")
