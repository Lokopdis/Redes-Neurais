import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Análisar os dados presentar na tabela:



# Carregar o dataset
arquivo = pd.read_csv('Eduardo_Turcatto_AVA_1/Dados/diabetes_prediction_dataset.csv')

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
accuracy = accuracy_score(y_test, y_pred)*100

# Mapeamento de texto para valores numéricos
map_text_to_num = {'Sim': 1, 'Não': 0}

# Função para prever diabetes
def predict_diabetes():
    try:
        novos_dados = {
            'gender': gender_var.get(),
            'age': float(age_var.get()),
            'hypertension': map_text_to_num[hypertension_var.get()],
            'heart_disease': map_text_to_num[heart_disease_var.get()],
            'smoking_history': smoking_history_var.get(),
            'bmi': float(bmi_var.get()),
            'HbA1c_level': float(HbA1c_level_var.get()),
            'blood_glucose_level': float(blood_glucose_level_var.get())
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
        resultado = 'Diabetes Detected' if previsao[0] == 1 else 'No Diabetes'
        messagebox.showinfo("Prediction Result", f'Prediction: {resultado}')
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Criar a janela principal
root = tk.Tk()
root.title("Será que você tem diabtes?")

# Criar o canvas
canvas = tk.Canvas(root, width=500, height=400)
canvas.pack()

# Labels e entradas para os inputs
tk.Label(root, text="Gênero").place(x=20, y=20)
gender_var = tk.StringVar()
gender_menu = ttk.Combobox(root, textvariable=gender_var, values=['Male', 'Female'])
gender_menu.place(x=150, y=20)

tk.Label(root, text="Idade").place(x=20, y=60)
age_var = tk.StringVar()
age_entry = tk.Entry(root, textvariable=age_var)
age_entry.place(x=150, y=60)

tk.Label(root, text="Hipertensão").place(x=20, y=100)
hypertension_var = tk.StringVar()
hypertension_menu = ttk.Combobox(root, textvariable=hypertension_var, values=['Não', 'Sim'])
hypertension_menu.place(x=150, y=100)

tk.Label(root, text="Doença Cardíaca").place(x=20, y=140)
heart_disease_var = tk.StringVar()
heart_disease_menu = ttk.Combobox(root, textvariable=heart_disease_var, values=['Não', 'Sim'])
heart_disease_menu.place(x=150, y=140)

tk.Label(root, text="Fum?").place(x=20, y=180)
smoking_history_var = tk.StringVar()
smoking_history_menu = ttk.Combobox(root, textvariable=smoking_history_var, values=['never', 'No Info', 'former', 'current', 'ever', 'not current'])
smoking_history_menu.place(x=150, y=180)

tk.Label(root, text="BMI").place(x=20, y=220)
bmi_var = tk.StringVar()
bmi_entry = tk.Entry(root, textvariable=bmi_var)
bmi_entry.place(x=150, y=220)

tk.Label(root, text="Nível HbA1c").place(x=20, y=260)
HbA1c_level_var = tk.StringVar()
HbA1c_level_entry = tk.Entry(root, textvariable=HbA1c_level_var)
HbA1c_level_entry.place(x=150, y=260)

tk.Label(root, text="Nível de glicose no sangue").place(x=20, y=300)
blood_glucose_level_var = tk.StringVar()
blood_glucose_level_entry = tk.Entry(root, textvariable=blood_glucose_level_var)
blood_glucose_level_entry.place(x=150, y=300)

# Botão para prever
predict_button = tk.Button(root, text="Prever", command=predict_diabetes)
predict_button.place(x=150, y=340)

# Rótulo para exibir a acurácia
accuracy_label = tk.Label(root, text=f"Esse modelo tem uma presição de: {accuracy:.2f}%", anchor='w')
accuracy_label.place(x=10, y=370)

# Executar o loop principal
root.mainloop()
